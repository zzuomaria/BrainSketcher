import math, sys
import torch
import sc_mbm.utils as ut

import math
inf = math.inf

import numpy as np
import time


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def train_one_epoch(model, data_loader, optimizer, device, epoch,
                    loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None,
                    img_feature_extractor=None, preprocess=None, global_step=0, text_embeddings=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    epoch_loss_accumulator = {
        'construction_loss': [],
        'emotion_loss': [],
        'category_loss': [],
        'classify_loss': [],
        'alignment_loss': [],
        'alignment_similarity': [],
        'final_loss': []
    }

    for data_iter_step, (data_dict) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_dict['eeg']
        emotion_labels = data_dict['emotion']
        category_labels = data_dict['category']

        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            images = data_dict['image']
            valid_idx = torch.nonzero(images.sum(dim=(1, 2, 3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']
        samples = samples.to(device)
        emotion_labels = emotion_labels.to(device)
        category_labels = category_labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _, classify_pred, emotion_emb = model(
                samples, img_features, valid_idx=valid_idx,
                mask_ratio=config.mask_ratio,
                classify_target_labels=[emotion_labels, category_labels],
                mode='train', text_embeddings=text_embeddings)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        if hasattr(model, 'module'):
            loss_cache = model.module.loss_cache
        else:
            loss_cache = model.loss_cache
        for k in epoch_loss_accumulator.keys():
            if k in loss_cache:
                epoch_loss_accumulator[k].append(loss_cache[k])

        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        pred = model_without_ddp.unpatchify(pred)
        cor = torch.mean(torch.tensor(
            [torch.corrcoef(torch.cat([p[0].unsqueeze(0), s[0].unsqueeze(0)], axis=0))[0, 1]
             for p, s in zip(pred, samples)]
        )).item()
        optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor)
        global_step += 1
        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print('train_loss_step:', np.mean(total_loss), 'lr:', lr, 'cor', np.mean(total_cor))

    if config.local_rank == 0:
        print(f"Epoch {epoch} training loss components:")
        for k, value in epoch_loss_accumulator.items():
            if value:
                avg_loss = sum(value) / len(value)
                print(f"  {k}: {avg_loss:.6f}\n")

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        epoch_metrics = {
            'epoch/train_loss': np.mean(total_loss),
            'epoch/cor': np.mean(total_cor),
            'epoch/lr': lr,
        }
        for k, v in epoch_loss_accumulator.items():
            if v:
                epoch_metrics[f'epoch/{k}'] = sum(v) / len(v)

        if start_time is not None:
            epoch_metrics['epoch/time_minutes'] = (time.time() - start_time) / 60.0
            log_writer.log_losses(epoch_metrics, step=epoch)
            print(f'epoch metrics logged to SwanLab at epoch {epoch}')

    if config.local_rank == 0:
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_cor), global_step
