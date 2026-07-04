# stage2_test.py
# 第二阶段测试代码：加载训练好的模型，批量生成图像
# Supports SketchSemanticProjector default path and optional CrossAttentionFusion legacy path
# Default use_fusion=False path: sketch CLS token -> two-layer MLP -> SDXL cross-attention
#
# ✅ 本次修改：生成/对比/草图/目标 的文件名包含【情感 + 类别 + ID】
# 例如：happy_castle_n02691156_542.png

import os
import sys
import argparse
import datetime
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/home/dell/dreamdiffusion_project/DreamDiffusion/code')

from eeg_ldm_refining import (
    init_eeg_encoder,
    init_sketch_encoder,
    CrossAttentionFusion,
    SemanticProjector,
    SketchSemanticProjector,
)
from stage2_model import EEGSketchToImageModel
from dataset import create_EEG_dataset

# 尝试从dataset中导入类别/情感列表（如果你的项目里有）
from dataset import CATEGORY_LIST, EMOTION_LIST



# ==================== 配置类 ====================

class TestConfig:
    """测试配置"""
    def __init__(self):
        # ========== 路径配置 ==========
        # 训练好的二阶段模型checkpoint路径
        self.checkpoint_path = "/data/DreamDiffusion/results/stage2/20251216_164532/checkpoints/best_model.pth"
        
        # EEG Encoder checkpoint（第一阶段预训练模型）- 与训练代码一致
        self.eeg_checkpoint = "/data/dreamdiffusion/output/results/eeg_pretrain/24-12-2025-23-53-32/checkpoints/checkpoint.pth"
        
        # 预训练模型路径
        self.clip_model = "/home/dell/dreamdiffusion_project/DreamDiffusion/pretrains/clip-vit-large-patch14"
        self.adapter_path = "/home/dell/image_generation/t2i-adapter-sketch-sdxl-1.0"
        self.sdxl_base_path = "/home/dell/image_generation/stable-diffusion-xl-base-1.0"
        self.vae_path = "/data/sdxl-vae-fp16-fix"
        
        # 数据路径

        #self.eeg_signals_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_images_alpha_raw_sub049.pth"
        self.eeg_signals_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_images_alpha_raw_sub033.pth"
        #self.eeg_signals_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_images_alpha_raw_sub003.pth"
        #self.eeg_signals_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/eeg_images_alpha_raw_sub049_clipped.pth"
        self.splits_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/train_val_test_sibling_group4.pth"
        #self.splits_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/train_val_test_sibling_group123.pth"
        
        # 输出路径
        self.output_dir = "/data/DreamDiffusion/results/stage2_test"
        
        # ========== 模型参数（需要和训练时一致）==========
        self.emotion_dim = 512       # emotion embedding 输入到fusion的维度
        self.sketch_dim = 1024       # sketch embedding 维度
        self.hidden_dim = 1024       # 融合模块隐藏层维度
        self.fusion_num_heads = 8    # 融合模块注意力头数
        
        # EmotionAdapter参数
        self.emotion_adapter_dim = 768  # EmotionAdapter的emotion_dim（来自MultiScaleEmotionEncoder输出）
        
        # ========== 测试参数 ==========
        self.batch_size = 1          # 测试时用1方便一一对应保存
        self.num_inference_steps = 50  # 去噪步数
        self.guidance_scale = 7.5    # CFG scale
        self.use_fusion = False      # default: sketch CLS -> MLP condition, no EEG-sketch fusion
        
        # ========== 输出选项 ==========
        self.save_generated = True   # 保存生成图像
        self.save_comparison = True  # 保存对比图（草图|生成|目标）
        self.save_sketch = True      # 单独保存草图
        self.save_target = True      # 单独保存目标图
        self.save_metadata = True    # 保存元数据（情感、类别等）
        
        # ========== 其他 ==========
        self.seed = 42
        self.num_workers = 4
        self.subject = 0             # 测试的受试者编号
        self.max_samples = None      # 最大测试样本数（None表示全部）
        self.test_on_train = False   # 是否在训练集上测试
    
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    def __str__(self):
        return '\n'.join(f'{k}: {v}' for k, v in self.__dict__.items())
    
    def to_dict(self):
        return self.__dict__.copy()


# ==================== 辅助函数 ====================

def normalize(img):
    """图像归一化到[-1, 1]"""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
        if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
            img = img.permute(2, 0, 1)
    return img * 2.0 - 1.0


def denormalize(img):
    """图像反归一化从[-1, 1]到[0, 1]"""
    return (img + 1) / 2


def load_sketch_images(sketch_paths, device):
    """加载草图图像"""
    sketch_images = []
    for path in sketch_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        sketch_images.append(img_tensor)
    return torch.stack(sketch_images).to(device)


def save_image(image_tensor, save_path, is_normalized=False):
    """
    保存单张图像
    Args:
        image_tensor: [3, H, W] 或 [H, W, 3]
        save_path: 保存路径
        is_normalized: 是否已归一化到[-1, 1]，如果是则先反归一化
    """
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
        # [3, H, W] -> [H, W, 3]
        image_tensor = image_tensor.permute(1, 2, 0)
    
    if is_normalized:
        image_tensor = denormalize(image_tensor)
    
    image_tensor = image_tensor.clamp(0, 1)
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(img_np).save(save_path)


def create_comparison_image(generated, sketch_path, target, output_size=512):
    """
    创建对比图：草图 | 生成图 | 目标图
    Args:
        generated: [3, H, W] 范围[0, 1]
        sketch_path: 草图路径
        target: [3, H, W] 范围[-1, 1]
        output_size: 每张图的大小
    Returns:
        PIL.Image: 拼接后的对比图
    """
    # 加载草图
    sketch = Image.open(sketch_path).convert("RGB").resize((output_size, output_size))
    
    # 生成图 [3, H, W] -> PIL
    gen_np = (generated.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gen_img = Image.fromarray(gen_np).resize((output_size, output_size))
    
    # 目标图 [3, H, W] (范围[-1,1]) -> PIL
    target_denorm = denormalize(target).clamp(0, 1)
    target_np = (target_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    target_img = Image.fromarray(target_np).resize((output_size, output_size))
    
    # 拼接：草图 | 生成 | 目标
    comparison = Image.new('RGB', (output_size * 3, output_size))
    comparison.paste(sketch, (0, 0))
    comparison.paste(gen_img, (output_size, 0))
    comparison.paste(target_img, (output_size * 2, 0))
    
    return comparison


def get_batch_metadata(batch):
    """从batch中提取元数据"""
    metadata = {}
    
    possible_fields = [
        'emotion', 'emotion_idx',
        'class', 'category', 'category_idx',
        'caption', 'text', 'label',
        'image_path', 'sketch_path'
    ]
    
    for field in possible_fields:
        if field in batch:
            value = batch[field]
            if isinstance(value, torch.Tensor):
                metadata[field] = value.cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)):
                metadata[field] = list(value)
            else:
                metadata[field] = value
    
    return metadata


def _safe_filename(s: str) -> str:
    """把任意字符串转成安全文件名"""
    s = str(s)
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']:
        s = s.replace(ch, '_')
    s = s.replace(' ', '_')
    return s


def build_base_filename(batch, i: int, sketch_path: str) -> str:
    """
    生成你想要的：emotion_category_imageID
    - emotion / category 优先从 batch 里的 idx 或字符串取
    - imageID 从 sketch 文件名解析（去掉类别前缀）
    """
    sketch_filename = os.path.basename(sketch_path)               # e.g. castle_n02691156_542.png
    base_no_ext = os.path.splitext(sketch_filename)[0]            # e.g. castle_n02691156_542

    # -------- category name / idx --------
    category_name = None
    category_idx = None

    if 'category_idx' in batch:
        try:
            category_idx = int(batch['category_idx'][i].item())
        except Exception:
            try:
                category_idx = int(batch['category_idx'][i])
            except Exception:
                category_idx = None

    if 'category' in batch and category_name is None:
        # 可能是字符串或列表
        try:
            v = batch['category'][i]
            category_name = v if isinstance(v, str) else str(v)
        except Exception:
            pass

    if category_name is None and CATEGORY_LIST is not None and category_idx is not None:
        if 0 <= category_idx < len(CATEGORY_LIST):
            category_name = CATEGORY_LIST[category_idx]

    # 兜底：从文件名取第一个token作为category
    if category_name is None:
        category_name = base_no_ext.split('_')[0]  # e.g. castle

    category_name = _safe_filename(category_name)

    # -------- emotion name / idx --------
    emotion_name = None
    emotion_idx = None

    if 'emotion_idx' in batch:
        try:
            emotion_idx = int(batch['emotion_idx'][i].item())
        except Exception:
            try:
                emotion_idx = int(batch['emotion_idx'][i])
            except Exception:
                emotion_idx = None

    if 'emotion' in batch and emotion_name is None:
        try:
            v = batch['emotion'][i]
            emotion_name = v if isinstance(v, str) else str(v)
        except Exception:
            pass

    if emotion_name is None and EMOTION_LIST is not None and emotion_idx is not None:
        if 0 <= emotion_idx < len(EMOTION_LIST):
            emotion_name = EMOTION_LIST[emotion_idx]

    # 兜底：没有的话用 emotion{idx} / emotionUnknown
    if emotion_name is None:
        emotion_name = f"emotion{emotion_idx}" if emotion_idx is not None else "emotionUnknown"

    emotion_name = _safe_filename(emotion_name)

    # -------- image id string from sketch filename --------
    # 规则与你给的参考一致：去掉 "{category_name}_"
    # 注意：category_name 可能做过safe替换，不影响长度，但我们用“原始文件名里解析出来的category token”更稳
    raw_category_token = base_no_ext.split('_')[0]
    if base_no_ext.startswith(raw_category_token + "_"):
        image_id_string = base_no_ext[len(raw_category_token) + 1:]
    else:
        # 兜底：去掉第一个下划线之前
        parts = base_no_ext.split('_', 1)
        image_id_string = parts[1] if len(parts) > 1 else base_no_ext

    image_id_string = _safe_filename(image_id_string)

    return f"{emotion_name}_{category_name}_{image_id_string}"


# ==================== 模型创建函数 ====================

def create_model(config, device):
    """创建并加载模型"""
    print("\n" + "="*70)
    print("创建模型...")
    print("="*70)
    
    # EEG Encoder
    print(f"加载EEG Encoder: {config.eeg_checkpoint}")
    eeg_encoder = init_eeg_encoder(config.eeg_checkpoint, device)
    
    # Sketch Encoder - 修复：添加device参数
    print("加载Sketch Encoder...")
    sketch_encoder = init_sketch_encoder(None, device)
    
    if config.use_fusion:
        print(f"Create fusion module (emotion_dim={config.emotion_dim}, sketch_dim={config.sketch_dim})")
        fusion_module = CrossAttentionFusion(
            emotion_dim=config.emotion_dim,
            sketch_dim=config.sketch_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.fusion_num_heads
        ).to(device)
        print("Create SemanticProjector for fused EEG-sketch features...")
        projector = SemanticProjector(
            input_dim=config.hidden_dim,
            hidden_dim=2048,
            pooled_dim=1280
        ).to(device)
    else:
        print("Create SketchSemanticProjector: sketch CLS -> MLP -> SDXL cross-attention condition...")
        fusion_module = None
        projector = SketchSemanticProjector(
            input_dim=config.sketch_dim,
            hidden_dim=2048,
            pooled_dim=1280
        ).to(device)
    
    # 主模型
    print("创建主模型 EEGSketchToImageModel...")
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
        emotion_dim=config.emotion_adapter_dim  # EmotionAdapter的维度
    )
    
    # 加载训练好的权重
    print(f"\n加载checkpoint: {config.checkpoint_path}")
    checkpoint = model.load_checkpoint(config.checkpoint_path)
    
    print(f"✅ 加载的是第 {checkpoint['epoch']} 个epoch的模型")
    if 'loss' in checkpoint:
        print(f"   该checkpoint的loss: {checkpoint['loss']:.4f}")
    if 'w_emotion' in checkpoint:
        print(f"   w_emotion: {checkpoint['w_emotion']}")
    if 'w_edge' in checkpoint:
        print(f"   w_edge: {checkpoint['w_edge']}")
    
    # 设置为评估模式
    if model.fusion_module is not None:
        model.fusion_module.eval()
    model.projector.eval()
    model.emotion_adapter.eval()
    
    return model, checkpoint


# ==================== 主测试函数 ====================

@torch.no_grad()
def test(config):
    """主测试流程"""
    
    print("\n" + "="*70)
    print("第二阶段测试：EEG + Sketch → Image")
    print("使用 CLIP分类 + SDXL文本编码 + EmotionAdapter")
    print("="*70)
    
    # 检查checkpoint路径
    if config.checkpoint_path is None or config.checkpoint_path == "":
        raise ValueError("必须指定 checkpoint_path！请在TestConfig中设置或通过 --checkpoint_path 参数指定")
    
    if not os.path.exists(config.checkpoint_path):
        print(f"⚠️  Checkpoint路径不存在: {config.checkpoint_path}")
        print("请检查路径是否正确，或者修改TestConfig中的checkpoint_path")
        raise FileNotFoundError(f"Checkpoint不存在: {config.checkpoint_path}")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    subdirs = {
        'generated': os.path.join(output_dir, "generated"),
        'comparison': os.path.join(output_dir, "comparison"),
        'sketch': os.path.join(output_dir, "sketch"),
        'target': os.path.join(output_dir, "target"),
    }
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    # 保存配置
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False, default=str)
    print(f"配置已保存到: {config_path}")
    
    # ========== 加载数据 ==========
    print("\n" + "-"*70)
    print("加载测试数据...")
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
        subject=config.subject
    )
    
    # 选择测试集
    if config.test_on_train:
        test_dataset = dataset_train
        print("⚠️ 在训练集上进行测试")
    else:
        test_dataset = dataset_test
        print("在测试集上进行测试")
    
    # 限制样本数
    if config.max_samples is not None and config.max_samples < len(test_dataset):
        print(f"限制测试样本数: {config.max_samples}")
        from torch.utils.data import Subset
        indices = list(range(config.max_samples))
        test_dataset = Subset(test_dataset, indices)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    print(f"批次数: {len(test_loader)}")
    
    # ========== 创建模型 ==========
    model, checkpoint = create_model(config, device)
    
    # ========== 开始生成 ==========
    print("\n" + "-"*70)
    print("开始生成图像...")
    print(f"去噪步数: {config.num_inference_steps}")
    print(f"CFG引导强度: {config.guidance_scale}")
    print(f"使用融合: {config.use_fusion}")
    print("-"*70)
    
    total_samples = len(test_dataset)
    all_metadata = []
    
    pbar = tqdm(test_loader, desc="生成中", total=len(test_loader))
    
    for idx, batch in enumerate(pbar):
        # 准备数据
        eeg = batch['eeg'].to(device)
        sketch_paths = batch['sketch_path']
        sketch_images = load_sketch_images(sketch_paths, device)
        target_images = batch['image']  # [B, 3, H, W]
        
        # 生成图像
        generated = model.generate(
            eeg,
            sketch_images,
            num_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            seed=config.seed
        )  # [B, 3, 1024, 1024]
        
        # 处理每个样本
        for i in range(generated.size(0)):
            sample_idx = idx * config.batch_size + i

            # ✅ 新命名：emotion_category_imageID
            base_filename = build_base_filename(batch, i, sketch_paths[i])

            # 保存生成的图像
            if config.save_generated:
                gen_path = os.path.join(subdirs['generated'], f"{base_filename}.png")
                save_image(generated[i], gen_path)
            
            # 保存对比图
            if config.save_comparison:
                comparison = create_comparison_image(
                    generated[i], sketch_paths[i], target_images[i]
                )
                comp_path = os.path.join(subdirs['comparison'], f"{base_filename}_comparison.png")
                comparison.save(comp_path)
            
            # 保存草图
            if config.save_sketch:
                sketch_img = Image.open(sketch_paths[i]).convert("RGB")
                sketch_save_path = os.path.join(subdirs['sketch'], f"{base_filename}_sketch.png")
                os.makedirs(os.path.dirname(sketch_save_path), exist_ok=True)
                sketch_img.save(sketch_save_path)
            
            # 保存目标图
            if config.save_target:
                target_path = os.path.join(subdirs['target'], f"{base_filename}_target.png")
                save_image(target_images[i], target_path, is_normalized=True)
            
            # 收集元数据
            if config.save_metadata:
                sample_meta = {
                    'index': sample_idx,
                    'base_filename': base_filename,
                    'sketch_path': sketch_paths[i],
                }
                batch_meta = get_batch_metadata(batch)
                for key, values in batch_meta.items():
                    if isinstance(values, list) and len(values) > i:
                        sample_meta[key] = values[i]
                all_metadata.append(sample_meta)
        
        # 更新进度条
        pbar.set_postfix({'已生成': sample_idx + 1})
    
    # 保存元数据
    if config.save_metadata and all_metadata:
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False, default=str)
        print(f"元数据已保存到: {metadata_path}")
    
    # ========== 完成 ==========
    print("\n" + "="*70)
    print("✅ 测试完成！")
    print("="*70)
    print(f"共生成 {total_samples} 张图像")
    print(f"输出目录: {output_dir}")
    print(f"  - 生成图像: {subdirs['generated']}")
    print(f"  - 对比图: {subdirs['comparison']}")
    if config.save_sketch:
        print(f"  - 草图: {subdirs['sketch']}")
    if config.save_target:
        print(f"  - 目标图: {subdirs['target']}")
    
    return output_dir


# ==================== 单样本测试函数 ====================

@torch.no_grad()
def test_single_sample(config, eeg_data, sketch_path, save_path=None):
    """
    测试单个样本
    Args:
        config: 配置对象
        eeg_data: EEG数据 [1, channels, time_len] 或 [channels, time_len]
        sketch_path: 草图路径
        save_path: 保存路径（可选）
    Returns:
        generated: 生成的图像 [3, H, W]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model, _ = create_model(config, device)
    
    # 准备数据
    if eeg_data.dim() == 2:
        eeg_data = eeg_data.unsqueeze(0)
    eeg = eeg_data.to(device)
    
    sketch_images = load_sketch_images([sketch_path], device)
    
    # 生成
    generated = model.generate(
        eeg,
        sketch_images,
        num_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        seed=config.seed
    )
    
    # 保存
    if save_path:
        save_image(generated[0], save_path)
        print(f"✅ 图像已保存到: {save_path}")
    
    return generated[0]


# ==================== 命令行参数 ====================

def get_args_parser():
    parser = argparse.ArgumentParser('Stage2 Testing', add_help=True)
    
    # checkpoint路径（可选，如果TestConfig中已设置则不需要）
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='训练好的模型checkpoint路径（覆盖TestConfig中的设置）')
    
    # 路径参数
    parser.add_argument('--eeg_checkpoint', type=str, default=None,
                        help='EEG Encoder checkpoint路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--eeg_signals_path', type=str, default=None,
                        help='EEG数据路径')
    parser.add_argument('--splits_path', type=str, default=None,
                        help='数据划分路径')
    
    # 模型参数
    parser.add_argument('--emotion_dim', type=int, default=None,
                        help='融合模块的emotion维度')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='融合模块隐藏层维度')
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='去噪步数')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='CFG引导强度')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大测试样本数')
    
    # 开关参数
    parser.add_argument('--use_fusion', action='store_true',
                        help='使用EEG+Sketch融合（默认不使用）')
    parser.add_argument('--test_on_train', action='store_true',
                        help='在训练集上测试')
    parser.add_argument('--no_comparison', action='store_true',
                        help='不保存对比图')
    parser.add_argument('--no_metadata', action='store_true',
                        help='不保存元数据')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--subject', type=int, default=0,
                        help='测试的受试者编号')
    
    return parser


# ==================== 入口点 ====================

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    config = TestConfig()
    
    # 从命令行参数更新配置
    config.update_from_args(args)
    
    # 处理开关参数
    if args.use_fusion:
        config.use_fusion = True
    if args.test_on_train:
        config.test_on_train = True
    if args.no_comparison:
        config.save_comparison = False
    if args.no_metadata:
        config.save_metadata = False
    
    # 打印配置
    print("\n" + "="*70)
    print("测试配置")
    print("="*70)
    print(config)
    
    # 执行测试
    test(config)
