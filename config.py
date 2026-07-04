import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass

class Config_MBM_EEG(Config_MAE_fMRI):
    """EEG MAE 预训练配置（含多尺度情感编码器与情感语义对齐）。"""
    def __init__(self):
        # 调试模式开关：开启后使用更小的epoch/batch/数据子集快速验证
        self.debug_mode = False

        # Training Parameters
        if self.debug_mode:
            self.lr = 2.5e-4
            self.min_lr = 0.
            self.weight_decay = 0.05
            self.num_epoch = 100
            self.warmup_epochs = 5
            self.batch_size = 32
            self.clip_grad = 0.8
        else:
            self.lr = 2.5e-4
            self.min_lr = 0.
            self.weight_decay = 0.05
            self.num_epoch = 300
            self.warmup_epochs = 75
            self.batch_size = 100
            self.clip_grad = 0.8

    # --------------------------------------------
    # MAE for fMRI
        # Model Parameters
        self.mask_ratio = 0.75
        self.patch_size = 4
        self.embed_dim = 1024
        self.decoder_embed_dim = 512
        self.depth = 24
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0

        # Project setting
        self.root_path = '/data/dreamdiffusion/output'
        self.output_path = '/data/dreamdiffusion/output/exps/'
        self.seed = 2022
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

        # 多尺度情感编码器（替换原情感分类头，输出768维情感嵌入）
        self.use_multi_scale_emotion = True

        # 情感语义对齐（EEG情感嵌入 ↔ CLIP文本嵌入，CLIP空间对比损失）
        self.use_semantic_alignment = True
        self.clip_dim = 768
        self.clip_model_path = '/home/dell/dreamdiffusion_project/DreamDiffusion/pretrains/clip-vit-large-patch14'
        self.alignment_temperature = 0.07

        # 调试模式：限制数据集大小
        if self.debug_mode:
            self.data_subset_ratio = 0.1
        else:
            self.data_subset_ratio = 1.0

class Config_EEG_finetune(Config_MBM_finetune):
    def __init__(self):

        # Project setting
        self.root_path = '/data/dreamdiffusion/output'
        # self.root_path = '.'
        self.output_path = '/data/dreamdiffusion/output/exps/'

        #self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        #self.eeg_signals_path = '/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/custom_eeg_data_from_npy.pth'

        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.splits_path = os.path.join(self.root_path, '/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/custom_block_splits.pth')

        self.dataset = 'EEG'
        self.pretrain_mbm_path = '../dreamdiffusion/pretrains/eeg_pretrain/checkpoint.pth'

        self.include_nonavg_test = True


        # Training Parameters
        self.lr = 5.3e-5
        self.weight_decay = 0.05
        self.num_epoch = 15
        self.batch_size = 16 if self.dataset == 'GOD' else 4
        self.mask_ratio = 0.5
        self.accum_iter = 1
        self.clip_grad = 0.8
        self.warmup_epochs = 2
        self.min_lr = 0.
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6

        # distributed training
        self.local_rank = 0

class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '/data/dreamdiffusion/output'
        self.output_path = '/data/dreamdiffusion/output/exps/'

        #self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        #self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')

        self.eeg_signals_path = os.path.join(self.root_path, '/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/custom_eeg_data_from_npy.pth')
        self.splits_path = os.path.join(self.root_path, '/home/dell/dreamdiffusion_project/DreamDiffusion/datasets/custom_block_splits.pth')

        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')

        self.dataset = 'EEG'
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 4 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 500

        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 1
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 4
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None
#增加两个
        self.clip_tune = True  # 假设默认是 True
        self.cls_tune = False # 假设默认是 False


class Config_Cls_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '/data/dreamdiffusion/output'
        self.output_path = '/data/dreamdiffusion/output/exps/'

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_14_70_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')

        self.dataset = 'EEG'
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 4 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 50

        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.15
        self.global_pool = False
        self.use_time_cond = False
        self.clip_tune = False
        self.subject = 1
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 4
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None
