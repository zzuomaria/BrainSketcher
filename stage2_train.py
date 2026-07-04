# stage2_train.py
# ç¬¬äºŒé˜¶æ®µè®­ç»ƒä»£ç ï¼šEEG + Sketch â†’ Image
# ä½¿ç”¨æ–°çš„CrossAttentionFusionèžåˆæ¨¡å— + EmotionAdapter æ³¨å…¥ + æƒ…æ„Ÿ/ç»“æž„æŸå¤±
#12/3 æ‹†åˆ†
#12/4 use_fusion å‚æ•°
import os
import sys
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
# æ·»åŠ é¡¹ç›®è·¯å¾„
sys.path.insert(0, '/home/dell/dreamdiffusion_project/DreamDiffusion/code')
# ä»Žeeg_ldm_refining.pyå¯¼å…¥æ¨¡å—
from eeg_ldm_refining import (
    init_eeg_encoder,
    init_sketch_encoder,
    CrossAttentionFusion,
    SemanticProjector,
    SketchSemanticProjector,
)

# ä»Žstage2_model.pyå¯¼å…¥ä¸»æ¨¡åž‹
from stage2_model import EEGSketchToImageModel
from dataset import create_EEG_dataset
# å¯é€‰ï¼šSwanLabæ—¥å¿—
try:
    import swanlab as sw
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("SwanLabæœªå®‰è£…ï¼Œå°†ä¸ä½¿ç”¨SwanLabæ—¥å¿—")
# ==================== é…ç½®ç±» ====================
class TrainConfig:
    """è®­ç»ƒé…ç½®"""
    def __init__(self):
        # è·¯å¾„
        self.eeg_checkpoint = "/data/dreamdiffusion/output/results/eeg_pretrain/24-12-2025-23-53-32/checkpoints/checkpoint.pth"
        #self.eeg_checkpoint = "/data/dreamdiffusion/output/results/eeg_pretrain/15-12-2025-23-51-54/checkpoints/checkpoint.pth"
        #self.eeg_checkpoint = "/data/dreamdiffusion/output/results/eeg_pretrain/08-12-2025-17-43-36/checkpoints/checkpoint.pth"
        
        
        self.clip_model = "/home/dell/dreamdiffusion_project/DreamDiffusion/pretrains/clip-vit-large-patch14"
        self.adapter_path = "/home/dell/image_generation/t2i-adapter-sketch-sdxl-1.0"
        self.sdxl_base_path = "/home/dell/image_generation/stable-diffusion-xl-base-1.0"
        self.vae_path = "/data/sdxl-vae-fp16-fix"
        #/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_images_alpha_raw_sub049_clipped.pth
        self.eeg_signals_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_images_alpha_raw_sub049_clipped.pth"
        #self.eeg_signals_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_sketches_alpha_raw_sub049.pth"
        self.splits_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/train_val_test_sibling_group123.pth"
        self.output_dir = "/data/DreamDiffusion/results/stage2"
        # è®­ç»ƒå‚æ•°
        self.num_epochs = 300
        self.batch_size = 4
        self.learning_rate = 5e-6 #1e-5
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        # å­¦ä¹ çŽ‡è°ƒåº¦
        self.lr_scheduler = "cosine"
        self.warmup_epochs = 20
        # æ¨¡åž‹å‚æ•°ï¼ˆèžåˆæ¨¡å—ï¼‰
        self.emotion_dim = 512
        self.sketch_dim = 1024
        self.hidden_dim = 1024
        self.fusion_num_heads = 8
        # æ–°å¢žï¼šæŸå¤±æƒé‡
        self.w_emotion = 0.1 #0.1
        self.w_edge = 0.5
        self.edge_loss_type = "l1"  # "l1" or "l2"
        # ä¿å­˜ä¸Žæ—¥å¿—
        self.save_every = 10
        self.val_every = 5
        self.log_every = 10
        self.generate_every = 20
        # å…¶ä»–
        self.seed = 42
        self.num_workers = 4
        self.resume_checkpoint = None
        # æ—¥å¿—
        self.use_swanlab = True
        self.project_name = "eeg_sketch_stage2_v2"
        # æ˜¯å¦èžåˆEEGï¼ŒFalseä»…ç”¨sketch
        self.use_fusion = False  # default: use sketch CLS MLP condition without EEG fusion
        self.gpu = '0'  #12/21æ–°å¢ž
    def update_from_args(self, args):
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    def __str__(self):
        return '\n'.join(f'{k}: {v}' for k, v in self.__dict__.items())
# ==================== æ•°æ®å¤„ç† ====================
def normalize(img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
        if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
            img = img.permute(2, 0, 1)
    return img * 2.0 - 1.0
def load_sketch_images(sketch_paths, device):
    """åŠ è½½è‰å›¾å›¾åƒ"""
    sketch_images = []
    for path in sketch_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        sketch_images.append(img_tensor)
    return torch.stack(sketch_images).to(device)
def save_images(images, save_dir, prefix, epoch):
    """ä¿å­˜ç”Ÿæˆçš„å›¾åƒ"""
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(save_dir, f"{prefix}_epoch{epoch}_{i}.png"))
# ==================== è®­ç»ƒå‡½æ•° ====================
def _build_emotion_prompts_from_batch(batch, fallback="an image with emotional tone"):
    """
    å°è¯•ä»Ž batch æž„é€ â€œæƒ…æ„Ÿ+ç±»åˆ«â€promptï¼š
    - ä¼˜å…ˆä½¿ç”¨ batch['caption'] æˆ– batch['text']
    - å…¶æ¬¡ä½¿ç”¨ batch['emotion'] + batch['class'] æ‹¼æŽ¥
    - å¦åˆ™ä½¿ç”¨ fallback
    è¿”å›ž List[str]ï¼Œé•¿åº¦ = B
    """
    B = len(batch['sketch_path'])
    prompts = []
    for i in range(B):
        if 'caption' in batch and isinstance(batch['caption'][i], str):
            prompts.append(batch['caption'][i])
        elif 'text' in batch and isinstance(batch['text'][i], str):
            prompts.append(batch['text'][i])
        elif ('emotion' in batch and 'class' in batch and
              isinstance(batch['emotion'][i], str) and isinstance(batch['class'][i], str)):
            prompts.append(f"a {batch['emotion'][i]} feeling {batch['class'][i]} image, expressive, high quality")
        else:
            prompts.append(fallback)
    return prompts

def train_one_epoch(model, train_loader, optimizer, device, config, epoch, logger=None):
    """è®­ç»ƒä¸€ä¸ªepoch"""
    if model.fusion_module is not None:
        model.fusion_module.train()
    model.projector.train()
    model.emotion_adapter.train()

    total_loss = 0
    num_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
    for batch_idx, batch in enumerate(pbar):
        # æ•°æ®
        eeg = batch['eeg'].to(device)
        sketch_images = load_sketch_images(batch['sketch_path'], device)
        target_images = batch['image'].to(device)
        # æž„é€ æƒ…æ„Ÿ prompts
        emotion_prompts = _build_emotion_prompts_from_batch(batch)

        # Forward
        #loss = model(
            #eeg, sketch_images, target_images,
            #emotion_text_prompts=emotion_prompts, edge_loss_type=config.edge_loss_type
        #)

        loss_dict = model(
            eeg, sketch_images, target_images,
            emotion_text_prompts=emotion_prompts, edge_loss_type=config.edge_loss_type
        )

        loss = loss_dict['total']

        # Backward
        #optimizer.zero_grad()
        #loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), config.max_grad_norm)
        #optimizer.step()

        #total_loss += loss.item()
        #pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), config.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({
            'total': f'{loss_dict["total"].item():.4f}',
            'diff': f'{loss_dict["diffusion"].item():.4f}',
            'emo': f'{loss_dict["emotion_weighted"].item():.4f}',
            'edge': f'{loss_dict["edge_weighted"].item():.4f}'
        })

        # æ—¥å¿—
        #if batch_idx % config.log_every == 0 and logger:
            #step = epoch * num_batches + batch_idx
            #logger.log({'train/loss': loss.item(), 'train/step': step})

        if batch_idx % config.log_every == 0 and logger:
                    step = epoch * num_batches + batch_idx
                    
                    # èŽ·å–å½“å‰çš„ alpha æƒé‡ (å‡è®¾æœ‰ 4 å±‚)
                    # alpha è¶ŠæŽ¥è¿‘ 0ï¼Œè¯´æ˜Žæ¨¡åž‹è¶Šä¾èµ– Sketch ç»“æž„ï¼›è¶ŠæŽ¥è¿‘ 1ï¼Œè¯´æ˜Žè¶Šä¾èµ– EEG æƒ…æ„Ÿæ³¨å…¥
                    alphas = model.emotion_adapter.get_alphas()
                    
                    log_dict = {
                        'train/loss_total': loss_dict['total'].item(),
                        'train/loss_diffusion': loss_dict['diffusion'].item(),
                        'train/loss_emotion': loss_dict['emotion'].item(),
                        'train/loss_edge': loss_dict['edge'].item(),
                        'train/step': step
                    }
                    
                    # å°†æ¯ä¸€å±‚çš„ alpha åŠ å…¥æ—¥å¿—
                    for i, a_val in enumerate(alphas):
                        log_dict[f'adapter/alpha_layer_{i}'] = a_val
                        
                    logger.log(log_dict)
        

    avg_loss = total_loss / num_batches
    return avg_loss
@torch.no_grad()

def validate(model, val_loader, device, config):
    """éªŒè¯"""
    if model.fusion_module is not None:
        model.fusion_module.eval()
    model.projector.eval()
    model.emotion_adapter.eval()
    total_loss = 0
    num_batches = len(val_loader)
    for batch in tqdm(val_loader, desc="Validating"):
        eeg = batch['eeg'].to(device)
        sketch_images = load_sketch_images(batch['sketch_path'], device)
        target_images = batch['image'].to(device)
        emotion_prompts = _build_emotion_prompts_from_batch(batch)
        
        #loss = model(
            #eeg, sketch_images, target_images,
            #emotion_text_prompts=emotion_prompts, edge_loss_type=config.edge_loss_type
        #)
        #total_loss += loss.item()

        loss_dict = model(
            eeg, sketch_images, target_images,
            emotion_text_prompts=emotion_prompts, edge_loss_type=config.edge_loss_type
        )
        total_loss += loss_dict['total'].item()

    return total_loss / num_batches
@torch.no_grad()

def generate_samples(model, val_loader, save_dir, epoch, num_samples=4):
    """ç”Ÿæˆç¤ºä¾‹å›¾åƒ"""
    if model.fusion_module is not None:
        model.fusion_module.eval()
    model.projector.eval()
    model.emotion_adapter.eval()

    device = model.device
    batch = next(iter(val_loader))
    eeg = batch['eeg'][:num_samples].to(device)
    sketch_paths = batch['sketch_path'][:num_samples]
    sketch_images = load_sketch_images(sketch_paths, device)
    target_images = batch['image'][:num_samples]
    #generated = model.generate(eeg, sketch_images, num_steps=30)
    generated = model.generate(
        eeg, sketch_images, 
        num_steps=50,           # å¢žåŠ æ­¥æ•°
        guidance_scale=7.5,     # CFGå¼•å¯¼
        seed=42                 # å›ºå®šç§å­ä¾¿äºŽå¯¹æ¯”
    )
    save_images(generated, save_dir, "generated", epoch)
    # ä¿å­˜è‰å›¾
    for i, path in enumerate(sketch_paths):
        img = Image.open(path).convert("RGB")
        img.save(os.path.join(save_dir, f"sketch_epoch{epoch}_{i}.png"))
    # ä¿å­˜ç›®æ ‡å›¾åƒ
    for i, img in enumerate(target_images):
        img_denorm = (img + 1) / 2
        img_np = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save(os.path.join(save_dir, f"target_epoch{epoch}_{i}.png"))
    print(f"âœ… ç¤ºä¾‹å›¾åƒå·²ä¿å­˜åˆ°: {save_dir}")
# ==================== ä¸»è®­ç»ƒæµç¨‹ ====================
def main(config):
    print("ç¬¬äºŒé˜¶æ®µè®­ç»ƒï¼šEEG + Sketch â†’ Image")
    print("ä½¿ç”¨ CrossAttentionFusion + EmotionAdapter æ³¨å…¥ + æƒ…æ„Ÿ/ç»“æž„æŸå¤±")
    # éšæœºç§å­
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    #12/21ä¿®æ”¹ï¼Œå¤šå¡è®­ç»ƒ
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU è®¾ç½®
    if ',' in config.gpu:
        # å¤šå¡ DDP
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        is_ddp = True
        print(f"ä½¿ç”¨å¤šå¡ DDP: {config.gpu}")
    else:
        device = torch.device(f'cuda:{config.gpu}')
        is_ddp = False
        print(f"ä½¿ç”¨å•å¡: cuda:{config.gpu}")

    print(f"è®¾å¤‡: {device}")
    # è¾“å‡ºç›®å½•
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    print(f"è¾“å‡ºç›®å½•: {output_dir}")
    # ä¿å­˜é…ç½®
    with open(os.path.join(output_dir, "config.txt"), 'w') as f:
        f.write(str(config))
    # æ—¥å¿—
    logger = None
    if config.use_swanlab and SWANLAB_AVAILABLE:
        sw.init(project=config.project_name, config=config.__dict__)
        logger = sw
        print("âœ… SwanLabæ—¥å¿—å·²åˆå§‹åŒ–")
    # ========== æ•°æ® ==========
    print("\n" + "-"*70)
    print("åŠ è½½æ•°æ®...")
    print("-"*70)

    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset_train, dataset_test = create_EEG_dataset(
        eeg_signals_path=config.eeg_signals_path,
        splits_path=config.splits_path,
        image_transform=img_transform,
        subject=0
    )

    #12/21ä¿®æ”¹ å¤šå¡è®­ç»ƒ
    #train_loader = DataLoader(
        #dataset_train, batch_size=config.batch_size,
        #shuffle=True, num_workers=config.num_workers, pin_memory=True
    #)
    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset_train, batch_size=config.batch_size,
        shuffle=shuffle, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        dataset_test, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    print(f"è®­ç»ƒé›†: {len(dataset_train)}")
    print(f"éªŒè¯é›†: {len(dataset_test)}")
    # ========== æ¨¡åž‹ ==========
    print("\n" + "-"*70)
    print("åˆ›å»ºæ¨¡åž‹...")
    print("-"*70)

    eeg_encoder = init_eeg_encoder(config.eeg_checkpoint, device)
    sketch_encoder = init_sketch_encoder(None, device)

    if config.use_fusion:
        fusion_module = CrossAttentionFusion(
            emotion_dim=config.emotion_dim,
            sketch_dim=config.sketch_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.fusion_num_heads
        ).to(device)
        projector = SemanticProjector(
            input_dim=config.hidden_dim,
            hidden_dim=2048,
            pooled_dim=1280
        ).to(device)
    else:
        fusion_module = None
        projector = SketchSemanticProjector(
            input_dim=config.sketch_dim,
            hidden_dim=2048,
            pooled_dim=1280
        ).to(device)
    model = EEGSketchToImageModel(
        eeg_encoder=eeg_encoder,
        sketch_encoder=sketch_encoder,
        fusion_module=fusion_module,
        projector=projector,
        adapter_path=config.adapter_path,
        sdxl_base_path=config.sdxl_base_path,
        vae_path=config.vae_path,
        device=device,
        use_fusion=config.use_fusion,
        w_emotion=config.w_emotion,
        w_edge=config.w_edge
    )
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        if model.fusion_module is not None:
            model.fusion_module = DDP(model.fusion_module, device_ids=[local_rank])
        model.projector = DDP(model.projector, device_ids=[local_rank])
        model.emotion_adapter = DDP(model.emotion_adapter, device_ids=[local_rank])

    # ========== ä¼˜åŒ–å™¨/è°ƒåº¦ ==========
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = config.num_epochs - config.warmup_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    # ========== æ¢å¤ ==========
    start_epoch = 0
    if config.resume_checkpoint:
        print(f"æ¢å¤è®­ç»ƒ: {config.resume_checkpoint}")
        ckpt = model.load_checkpoint(config.resume_checkpoint)
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt['scheduler']:
            scheduler.load_state_dict(ckpt['scheduler'])
        print(f"ä»Žepoch {start_epoch} ç»§ç»­è®­ç»ƒ")
    # ========== è®­ç»ƒå¾ªçŽ¯ ==========
    print("\n" + "-"*70)
    print("å¼€å§‹è®­ç»ƒ...")
    print("-"*70)
    best_val_loss = float('inf')
    for epoch in range(start_epoch, config.num_epochs):

        if is_ddp:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, config, epoch, logger
        )

        if epoch >= config.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{config.num_epochs} - Train Loss: {train_loss:.4f} - LR: {current_lr:.2e}")
        if logger:
            logger.log({'epoch': epoch, 'train/epoch_loss': train_loss, 'train/lr': current_lr})
        if (epoch + 1) % config.val_every == 0:
            val_loss = validate(model, val_loader, device, config)
            print(f"Validation Loss: {val_loss:.4f}")
            if logger:
                logger.log({'val/loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(
                    os.path.join(output_dir, "checkpoints", "best_model.pth"),
                    epoch, optimizer, scheduler, val_loss
                )
                print(f"âœ… ä¿å­˜æœ€ä½³æ¨¡åž‹ (val_loss: {val_loss:.4f})")
        if (epoch + 1) % config.save_every == 0:
            model.save_checkpoint(
                os.path.join(output_dir, "checkpoints", f"epoch_{epoch}.pth"),
                epoch, optimizer, scheduler, train_loss
            )

        if (epoch + 1) % config.generate_every == 0:
            generate_samples(
                model, val_loader,
                os.path.join(output_dir, "samples"),
                epoch, num_samples=4
            )

    model.save_checkpoint(
        os.path.join(output_dir, "checkpoints", "final_model.pth"),
        config.num_epochs - 1, optimizer, scheduler, train_loss
    )

    print("\n" + "="*70)
    print("è®­ç»ƒå®Œæˆï¼")
    print(f"æœ€ä½³éªŒè¯æŸå¤±: {best_val_loss:.4f}")
    print(f"è¾“å‡ºç›®å½•: {output_dir}")
    print("="*70)
    if logger:
        sw.finish()

# ==================== å‘½ä»¤è¡Œå‚æ•° ====================
def get_args_parser():
    parser = argparse.ArgumentParser('Stage2 Training', add_help=True)
    # è·¯å¾„
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    # è®­ç»ƒå‚æ•°
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    # æ¨¡åž‹å‚æ•°
    parser.add_argument('--emotion_dim', type=int, default=None)
    parser.add_argument('--sketch_dim', type=int, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--fusion_num_heads', type=int, default=None)
    # æ–°å¢žï¼šæŸå¤±æƒé‡å’Œè¾¹ç¼˜æŸå¤±ç±»åž‹
    parser.add_argument('--w_emotion', type=float, default=None)
    parser.add_argument('--w_edge', type=float, default=None)
    parser.add_argument('--edge_loss_type', type=str, default=None, choices=['l1', 'l2'])
    # ä¿å­˜ä¸Žæ—¥å¿—
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--val_every', type=int, default=None)
    parser.add_argument('--generate_every', type=int, default=None)
    # å…¶ä»–
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no_swanlab', action='store_true')
    # 12/4æ–°å¢žï¼Œåªç”¨è‰å›¾è®­ç»ƒ
    parser.add_argument('--no_fusion', action='store_true', help='åªç”¨sketchï¼Œä¸èžåˆEEG')
    # 12/21æ–°å¢žï¼Œå¤šå¡è®­ç»ƒ
    parser.add_argument('--gpu', type=str, default=None, help='GPU: "0", "1", or "0,1"')
    return parser
if __name__ == '__main__':
    args = get_args_parser().parse_args()

    config = TrainConfig()
    config.update_from_args(args)

    if args.no_swanlab:
        config.use_swanlab = False
    if args.no_fusion:
        config.use_fusion = False
    print("\né…ç½®:")
    print("-"*40)
    print(config)
    print("-"*40)

    main(config)

