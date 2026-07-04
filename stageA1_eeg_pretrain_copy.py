import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import timm.optim as timm_optim
import datetime
import matplotlib.pyplot as plt

import swanlab as sw
import copy

from config import Config_MBM_EEG
from dataset import eeg_pretrain_dataset, EMOTION_LIST
from sc_mbm.mae_for_eeg import MAEforEEG
from sc_mbm.trainer import train_one_epoch
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model
from transformers import CLIPModel, CLIPProcessor


class swanlab_logger:
    def __init__(self, config):
        sw.init(
            project="dreamdiffusion",
            config=config.__dict__,
        )
        self.config = config
        self.step = None

    def log(self, name, data, step=None):
        if step is None:
            sw.log({name: data})
        else:
            sw.log({name: data}, step=step)
            self.step = step

    def log_losses(self, loss_dict, step=None):
        if step is None:
            step = self.step
        sw.log(loss_dict, step=step)
        if step is not None:
            self.step = step

    def log_metrics(self, metrics_dict, prefix='train', step=None):
        if step is None:
            step = self.step
        log_dict = {f'{prefix}/{key}': value for key, value in metrics_dict.items()}
        sw.log(log_dict, step=step)
        if step is not None:
            self.step = step

    def log_image(self, name, fig):
        if self.step is None:
            sw.log({name: sw.Image(fig)})
        else:
            sw.log({name: sw.Image(fig)}, step=self.step)

    def finish(self):
        sw.finish()


def get_args_parser():
    parser = argparse.ArgumentParser('MBM pre-training for EEG', add_help=False)

    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--decoder_num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)
    parser.add_argument('--include_hcp', type=bool)
    parser.add_argument('--include_kam', type=bool)
    parser.add_argument('--use_nature_img_loss', type=bool)
    parser.add_argument('--img_recon_weight', type=float)

    # distributed training parameters
    parser.add_argument('--local_rank', type=int)

    # multi-scale emotion encoder
    parser.add_argument('--use_multi_scale_emotion', type=bool, default=True,
                        help='Whether to use multi-scale emotion encoder')
    return parser


def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def fmri_transform(x, sparse_rate=0.2):
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0] * sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def main(config):
    if config.debug_mode:
        print("=" * 50)
        print("RUNNING IN DEBUG MODE")
        print(f"Epochs: {config.num_epoch}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Dataset Ratio: {config.data_subset_ratio}")
        print(f"Warmup Epochs: {config.warmup_epochs}")
        print("=" * 50)

    if torch.distributed.is_available() and 'LOCAL_RANK' in os.environ:
        config.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(config.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://',
                                             world_size=int(os.environ['WORLD_SIZE']),
                                             rank=int(os.environ['RANK']))
    else:
        config.local_rank = 0

    output_path = os.path.join(config.root_path, 'results', 'eeg_pretrain',
                               '%s' % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path

    logger = swanlab_logger(config) if config.local_rank == 0 else None
    global_step = 0
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)

    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset_pretrain = eeg_pretrain_dataset(
        path='/data/sorted_images_eeg_alpha_raw_clipped_split/sub003_split/train',
        roi=config.roi, patch_size=config.patch_size,
        transform=fmri_transform, aug_times=config.aug_times, num_sub_limit=config.num_sub_limit,
        include_kam=config.include_kam, include_hcp=config.include_hcp)

    original_data_len = dataset_pretrain.data_len
    original_data_chan = dataset_pretrain.data_chan
    if hasattr(config, 'debug_mode') and config.debug_mode:
        subset_size = int(len(dataset_pretrain) * config.data_subset_ratio)
        indices = torch.randperm(len(dataset_pretrain))[:subset_size]
        dataset_pretrain = torch.utils.data.Subset(dataset_pretrain, indices)
        dataset_pretrain.data_len = original_data_len
        dataset_pretrain.data_chan = original_data_chan
        print(f'DEBUG MODE: Using {subset_size} samples')

    print(f'Dataset size: {len(dataset_pretrain)}\n Time len: {dataset_pretrain.data_len}')

    sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None

    dataloader_eeg = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler,
                                shuffle=(sampler is None), pin_memory=True)

    config.time_len = dataset_pretrain.data_len

    model = MAEforEEG(time_len=dataset_pretrain.data_len, patch_size=config.patch_size, embed_dim=config.embed_dim,
                      decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, in_chans=dataset_pretrain.data_chan,
                      num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                      focus_range=config.focus_range, focus_rate=config.focus_rate,
                      img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss,
                      use_multi_scale_emotion_encoder=config.use_multi_scale_emotion,
                      use_semantic_alignment=config.use_semantic_alignment,
                      clip_dim=config.clip_dim)

    model.to(device)
    model_without_ddp = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank,
                                        find_unused_parameters=config.use_nature_img_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.92, 0.95), weight_decay=config.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    cor_list = []
    start_time = time.time()
    print('Start Training the EEG MAE ... ...')

    # 情感语义对齐：加载CLIP并预计算情感文本embedding
    clip_model = None
    clip_processor = None
    text_embeddings = None
    if config.use_semantic_alignment:
        print('=' * 50)
        print("loading CLIP model for semantic alignment")
        try:
            clip_model = CLIPModel.from_pretrained(config.clip_model_path).to(device)
            clip_processor = CLIPProcessor.from_pretrained(config.clip_model_path)
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad = False
            clip_dim = clip_model.text_projection.out_features
            print(f"CLIP model loaded, text dim={clip_dim}, config dim={config.clip_dim}")
            assert clip_dim == config.clip_dim, f"CLIP dim mismatch: model={clip_dim}, config={config.clip_dim}"
        except Exception as e:
            print(f"CLIP model load failed: {e}")
            sys.exit(1)

        if clip_model is not None and clip_processor is not None:
            print("pre-computing emotion text embeddings")
            with torch.no_grad():
                text_embeddings = model_without_ddp.semantic_alignment.get_text_embeddings(
                    EMOTION_LIST, clip_model, clip_processor
                )
                text_embeddings = text_embeddings.to(device)
            print(f"text embeddings done: num_emotions={len(EMOTION_LIST)}, shape={text_embeddings.shape}")
            print(f"emotion list: {EMOTION_LIST}")
            print("=" * 70 + "\n")

    # 图像特征提取器（use_nature_img_loss开启时使用）
    img_feature_extractor = None
    preprocess = None
    if config.use_nature_img_loss:
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.models.feature_extraction import create_feature_extractor
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        m = resnet50(weights=weights)
        img_feature_extractor = create_feature_extractor(m, return_nodes={f'layer2': 'layer2'}).to(device).eval()
        for param in img_feature_extractor.parameters():
            param.requires_grad = False

    for ep in range(config.num_epoch):
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(ep)
        cor, global_step = train_one_epoch(
            model, dataloader_eeg, optimizer, device, ep, loss_scaler,
            logger, config, start_time, model_without_ddp,
            img_feature_extractor, preprocess, global_step, text_embeddings=text_embeddings)
        cor_list.append(cor)

        save_interval = 1 if config.debug_mode else 20
        if (ep % save_interval == 0 or ep + 1 == config.num_epoch) and config.local_rank == 0:
            save_model(config, ep, model_without_ddp, optimizer, loss_scaler,
                       os.path.join(output_path, 'checkpoints'))
            plot_recon_figures(model, device, dataset_pretrain, output_path, 5,
                               config, logger, model_without_ddp)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        final_metrics = {
            'final/max_cor': np.max(cor_list),
            'final/total_time': total_time,
            'final/total_epochs': config.num_epoch,
            'final/total_steps': global_step
        }
        if hasattr(model, 'module'):
            loss_cache = model.module.loss_cache
        else:
            loss_cache = model.loss_cache if hasattr(model, 'loss_cache') else {}
        for k in ['construction_loss', 'emotion_loss', 'category_loss', 'classify_loss', 'final_loss']:
            if k in loss_cache:
                final_metrics[f'final/{k}'] = np.mean(loss_cache[k])
        logger.log_losses(final_metrics, step=config.num_epoch)
        logger.finish()
    return


@torch.no_grad()
def plot_recon_figures(model, device, dataset, output_path, num_figures=5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30, 15))
    fig.tight_layout()
    axs[0, 0].set_title('Ground-truth')
    axs[0, 1].set_title('Masked Ground-truth')
    axs[0, 2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['eeg']
        sample = sample.to(device)
        _, pred, mask, classify_pred, emotion_emb = model(sample, mask_ratio=config.mask_ratio, mode='eval')
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        mask = mask.to('cpu').numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0, 1]

        x_axis = np.arange(0, sample.shape[-1])
        ax[0].plot(x_axis, sample)
        s = 0
        for x, m in zip(sample_with_mask, mask):
            if m == 0:
                ax[1].plot(x_axis[s:s + len(x)], x, color='#1f77b4')
            s += len(x)
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f' % cor, weight='bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s' % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)


def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_EEG()
    config = update_config(args, config)
    main(config)
