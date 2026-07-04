import sys
sys.path.append('../dreamdiffusion/code/')
import sc_mbm.utils as ut
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
from dataset import CATEGORY_LIST, EMOTION_LIST


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean', device="cuda"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.tensor(alpha, device=device)

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).type_as(inputs)
        log_probs = torch.sum(log_probs * targets_one_hot, dim=1)
        probs = torch.exp(log_probs)
        alpha = torch.sum(self.alpha * targets_one_hot, dim=1)
        focal_loss = -alpha * ((1 - probs) ** self.gamma) * log_probs
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassifyNet(nn.Module):
    def __init__(self, dim_list):
        super(ClassifyNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dim_list) - 1):
            self.layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
            if i < len(dim_list) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PatchEmbed1D(nn.Module):
    """1D version of data (fmri voxels) to Patch Embedding"""
    def __init__(self, time_len=224, patch_size=1, in_chans=64, embed_dim=256):
        super().__init__()
        num_patches = time_len // patch_size
        self.patch_shape = patch_size
        self.time_len = time_len
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape
        x = self.proj(x).transpose(1, 2).contiguous()
        return x


class MultiScaleEmotionEncoder(nn.Module):
    """多尺度情感编码器 - 在不同时间尺度上提取EEG情感特征。

    通过并行的多尺度池化 + 多头自注意力，捕捉局部与全局的情感表征，
    融合后投影到CLIP文本嵌入空间（768维），用于后续与CLIP情感描述做语义对齐。
    """
    def __init__(self, embed_dim=1024, num_heads=8, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim

        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_scales)
        ])

        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool1d(output_size) for output_size in [64, 32, 16]
        ])

        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.emotion_projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 768),
            nn.LayerNorm(768)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, embed_dim] - patch tokens (不含cls_token)
        Returns:
            emotion_emb: [batch, 768] - 与CLIP文本空间对齐的情感特征
        """
        batch_size, seq_len, embed_dim = x.size()
        x_transposed = x.transpose(1, 2)

        scale_features = []
        for attn, pool in zip(self.scale_attentions, self.pooling_layers):
            pooled = pool(x_transposed).transpose(1, 2)
            scale_out, _ = attn(pooled, pooled, pooled)
            scale_pooled = scale_out.mean(dim=1)
            scale_features.append(scale_pooled)

        multi_scale = torch.cat(scale_features, dim=-1)
        fused = self.scale_fusion(multi_scale)
        emotion_emb = self.emotion_projector(fused)
        return emotion_emb


class EmotionSemanticAlignment(nn.Module):
    """情感语义对齐模块 - 将EEG情感嵌入与CLIP情感文本嵌入对齐。

    使用一组情感描述模板，经CLIP文本编码器得到每个情感的文本嵌入；
    训练时EEG emotion_emb与文本嵌入做相似度计算，以交叉熵监督，
    使EEG情感表征在CLIP语义空间中靠近对应情感描述。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.emotion_text_templates = [
            "the emotion of {} is visible",
        ]

    def get_text_embeddings(self, emotion_names, clip_model, clip_processor):
        """预计算所有情感描述的CLIP文本embedding（多模板平均）。
        Returns: [num_emotions, clip_dim]，已L2归一化。
        """
        device = next(clip_model.parameters()).device
        all_embeddings = []
        for emotion in emotion_names:
            emotion_embeddings = []
            for template in self.emotion_text_templates:
                text = template.format(emotion)
                inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    text_features = clip_model.get_text_features(**inputs)
                    emotion_embeddings.append(text_features)
            avg_emb = torch.stack(emotion_embeddings).mean(dim=0)
            all_embeddings.append(avg_emb)
            result = torch.cat(all_embeddings, dim=0)
            result = F.normalize(result, p=2, dim=1)
        return result

    def forward(self, emotion_emb, emotion_labels, text_embeddings):
        """
        Args:
            emotion_emb: [batch, 768] EEG情感特征
            emotion_labels: [batch] 情感标签
            text_embeddings: [num_emotions, 768] 预计算的CLIP文本embedding
        Returns:
            alignment_loss: 标量
        """
        emotion_emb_norm = F.normalize(emotion_emb, p=2, dim=1)
        text_emb_norm = F.normalize(text_embeddings, p=2, dim=1)

        similarities = torch.matmul(emotion_emb_norm, text_emb_norm.t())
        similarities = similarities / self.temperature
        alignment_loss = F.cross_entropy(similarities, emotion_labels)
        return alignment_loss


class MAEforEEG(nn.Module):
    """Masked Autoencoder for EEG，含多尺度情感编码器与情感语义对齐模块。

    Args:
        classify_dims: 2-D list，0轴为分类任务数，1轴为每层维度；
            首尾需分别等于输入维度和分类类别数。
        use_multi_scale_emotion_encoder: 是否启用多尺度情感编码器（替换原情感分类头）。
        use_semantic_alignment: 是否启用与CLIP文本空间的情感语义对齐损失。
    """
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=64,
                 depth=24, num_heads=16, decoder_embed_dim=512,
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, focus_range=None, focus_rate=None, img_recon_weight=1.0,
                 use_nature_img_loss=False, use_classify=True,
                 classify_dims=[[64, len(EMOTION_LIST)], [64, len(CATEGORY_LIST)]],
                 use_multi_scale_emotion_encoder=True, use_semantic_alignment=True, clip_dim=768):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss
        self.use_classify = use_classify
        self.use_multi_scale_emotion_encoder = use_multi_scale_emotion_encoder
        self.use_semantic_alignment = use_semantic_alignment
        self.clip_dim = clip_dim

        # MAE encoder
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        num_patches = int(time_len / patch_size)
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.input_bn = nn.BatchNorm1d(in_chans)
        self.patch_bn = nn.BatchNorm1d(embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # MAE decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size, bias=True)
        self.output_bn = nn.BatchNorm1d(in_chans * patch_size)

        # nature image decoder (off by default)
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])
            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28*28, bias=True)
            )

        # 分类任务与多尺度情感编码器
        if use_classify:
            if use_multi_scale_emotion_encoder:
                self.multi_scale_emotion_encoder = MultiScaleEmotionEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_scales=3
                )
                emotion_input_dim = clip_dim
                self.emotion_classify_net = ClassifyNet([emotion_input_dim] + classify_dims[0][1:])
            else:
                self.classify_compress_layer = nn.Conv1d(2*in_chans+1, 2*in_chans+1,
                                                         kernel_size=patch_size*64, stride=patch_size*64)
                classify_dims[0][0] = (2*in_chans+1) * patch_size
                classify_dims[1][0] = (2*in_chans+1) * patch_size
                self.emotion_classify_net = ClassifyNet(classify_dims[0])

            if not use_multi_scale_emotion_encoder:
                self.category_classify_net = ClassifyNet(classify_dims[1])
            else:
                self.classify_compress_layer = nn.AdaptiveAvgPool1d(1)
                category_input_dim = embed_dim
                self.category_classify_net = ClassifyNet([category_input_dim] + classify_dims[1][1:])

            self.classify_nets = [self.emotion_classify_net, self.category_classify_net]
            self.classify_task_num = len(classify_dims)

        if use_semantic_alignment:
            self.semantic_alignment = EmotionSemanticAlignment()
            self.clip_model = None
            self.clip_processor = None
            self.text_embeddings = None
            self.emotion_names = EMOTION_LIST

        self.initialize_weights()
        self.loss_cache = {
            'construction_loss': 0.0,
            'emotion_loss': 0.0,
            'category_loss': 0.0,
            'classify_loss': 0.0,
            'alignment_loss': 0.0,
            'alignment_similarity': 0.0,
            'final_loss': 0.0
        }

    def initialize_weights(self):
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_nature_img_loss:
            nature_img_decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.nature_img_decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
            self.nature_img_decoder_pos_embed.data.copy_(torch.from_numpy(nature_img_decoder_pos_embed).float().unsqueeze(0))
            torch.nn.init.normal_(self.nature_img_mask_token, std=.02)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[1] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size
        h = x.shape[1]
        imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
        return imgs.transpose(1, 2)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1 - self.focus_rate] * L
            weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size
                        ] = [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)

        noise = torch.rand(N, L, device=x.device)
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i, :]] = 1.1

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x: [batch, in_chans, time_len]
        x = self.input_bn(x)
        x = self.patch_embed(x)  # [batch, num_patches, embed_dim]

        batch_size, num_patches, embed_dim = x.shape
        x_reshaped = x.reshape(-1, embed_dim)
        x_normalized = self.patch_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, num_patches, embed_dim)

        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore=None):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        batch_size, num_patches_plus_one, patch_dim = x.shape
        x_reshaped = x.reshape(-1, patch_dim)
        x_normalized = self.output_bn(x_reshaped)
        x = x_normalized.reshape(batch_size, num_patches_plus_one, patch_dim)

        x = x[:, 1:, :]
        return x

    def forward_nature_img_decoder(self, x, ids_restore):
        x = self.nature_img_decoder_embed(x)
        mask_tokens = self.nature_img_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.nature_img_decoder_pos_embed
        for blk in self.nature_img_decoder_blocks:
            x = blk(x)
        x = self.nature_img_decoder_norm(x)
        x = x[:, 1:, :]
        x = self.nature_img_decoder_pred(x)
        x = x.view(x.shape[0], 512, 28, 28)
        return x

    def forward_nature_img_loss(self, inputs, reconstructions):
        loss = ((torch.tanh(inputs) - torch.tanh(reconstructions)) ** 2).mean()
        if torch.isnan(reconstructions).sum():
            print('nan in reconstructions')
        if torch.isnan(inputs).sum():
            print('nan in inputs')
        return loss

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, in_chans, T]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove
        """
        imgs = imgs.transpose(1, 2)
        target = self.patchify(imgs)
        loss = torch.abs(pred - target)
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (loss * mask).sum()
        return loss

    def forward(self, imgs, img_features=None, valid_idx=None, mask_ratio=0.75,
                classify_target_labels=[], mode='train', text_embeddings=None):
        if not hasattr(self, 'loss_cache'):
            self.loss_cache = {}
        self.loss_cache.clear()

        emotion_emb = None

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        construction_loss = self.forward_loss(imgs, pred, mask)

        if self.use_nature_img_loss and img_features is not None:
            if len(valid_idx) != 0:
                nature_image_recon = self.forward_nature_img_decoder(latent[valid_idx], ids_restore[valid_idx])
                loss_nature_image_recon = self.forward_nature_img_loss(img_features, nature_image_recon)
                if torch.isnan(loss_nature_image_recon).sum():
                    print(loss_nature_image_recon)
                    print("loss_nature_image_recon is nan")
                construction_loss = construction_loss + self.img_recon_weight * loss_nature_image_recon

        assert mode != 'train' or len(classify_target_labels) == len(self.classify_nets)

        classify_preds = []
        classify_losses = []

        emotion_loss = torch.tensor(0.0, device=imgs.device)
        category_loss = torch.tensor(0.0, device=imgs.device)
        classify_loss = torch.tensor(0.0, device=imgs.device)
        alignment_loss = torch.tensor(0.0, device=imgs.device)

        if self.use_classify:
            if self.use_multi_scale_emotion_encoder:
                # 1. 提取patch tokens（去除cls_token）
                patch_tokens = latent[:, 1:, :]
                # 2. 多尺度情感编码，emotion_emb 传递给第二阶段
                emotion_emb = self.multi_scale_emotion_encoder(patch_tokens)
                # 3. 情感分类
                emotion_logit = self.emotion_classify_net(emotion_emb)
                emotion_pred = nn.Softmax(1)(emotion_logit)
                classify_preds.append(emotion_pred)
                # 4. 类别分类
                latent_transposed = latent.transpose(1, 2)
                classify_latent = self.classify_compress_layer(latent_transposed)
                classify_latent = classify_latent.squeeze(-1)
                category_logit = self.category_classify_net(classify_latent)
                category_pred = nn.Softmax(1)(category_logit)
                classify_preds.append(category_pred)
                # 5. 计算损失
                if mode == 'train':
                    alpha = torch.tensor([0.1500, 0.1839, 0.0932, 0.0658, 0.1173, 0.1945, 0.1291, 0.0664],
                                         device=emotion_logit.device)
                    emotion_targets = classify_target_labels[0]
                    emotion_loss = FocalLoss(alpha=alpha)(emotion_logit, emotion_targets)
                    category_targets = classify_target_labels[1]
                    category_loss = nn.CrossEntropyLoss()(category_logit, category_targets)
                    classify_loss = 1.0 * emotion_loss + 0.0 * category_loss  # 仅优化情感分类
                    classify_losses.append(classify_loss)
            else:
                classify_latent = self.classify_compress_layer(latent)
                classify_latent = torch.flatten(classify_latent, start_dim=1)
                for i, classify_model in enumerate(self.classify_nets):
                    logit = classify_model(classify_latent)
                    classify_pred = nn.Softmax(1)(logit)
                    classify_preds.append(classify_pred)
                    if mode == 'train':
                        if i == 0:
                            alpha = torch.tensor([0.1500, 0.1839, 0.0932, 0.0658, 0.1173, 0.1945, 0.1291, 0.0664],
                                                 device=logit.device)
                            emotion_targets = classify_target_labels[i]
                            focal_loss = FocalLoss(alpha=alpha, gamma=2)(logit, emotion_targets)
                            classify_losses.append(focal_loss * 0.7)
                        else:
                            class_targets = classify_target_labels[i]
                            class_loss = nn.CrossEntropyLoss()(logit, class_targets)
                            classify_losses.append(class_loss * 0.3)

        if self.use_semantic_alignment and mode == 'train' and emotion_emb is not None:
            if text_embeddings is not None:
                alignment_loss = self.semantic_alignment(
                    emotion_emb,
                    classify_target_labels[0],
                    text_embeddings
                )
                with torch.no_grad():
                    emotion_emb_norm = F.normalize(emotion_emb, p=2, dim=1)
                    text_emb_norm = F.normalize(text_embeddings, p=2, dim=1)
                    similarities = torch.matmul(emotion_emb_norm, text_emb_norm.t())
                    batch_indices = torch.arange(len(classify_target_labels[0]), device=imgs.device)
                    avg_similarity = similarities[batch_indices, classify_target_labels[0]].mean()
                    self.loss_cache['alignment_similarity'] = avg_similarity.item()
            else:
                alignment_loss = torch.tensor(0.0, device=imgs.device)

        if mode == 'train':
            loss = 1.0 * construction_loss + 2.0 * sum(classify_losses) + 1.0 * alignment_loss
            self.loss_cache['construction_loss'] = construction_loss.item()
            self.loss_cache['emotion_loss'] = emotion_loss.item()
            self.loss_cache['category_loss'] = category_loss.item()
            self.loss_cache['classify_loss'] = classify_loss.item()
            self.loss_cache['alignment_loss'] = alignment_loss.item()
            self.loss_cache['final_loss'] = loss.item()
        else:
            loss = construction_loss

        return loss, pred, mask, classify_preds, emotion_emb


class eeg_encoder(nn.Module):
    """MAE编码器的独立版本，用于第二阶段特征提取。
    与 MAEforEEG 的编码器、多尺度情感编码器保持一致，便于加载预训练权重。
    """
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=64,
                 depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=False,
                 use_classify=True,
                 classify_dims=[[64, len(EMOTION_LIST)], [64, len(CATEGORY_LIST)]],
                 use_multi_scale_emotion_encoder=True):
        super().__init__()

        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)
        num_patches = int(time_len / patch_size)
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool

        self.use_classify = use_classify
        self.use_multi_scale_emotion_encoder = use_multi_scale_emotion_encoder

        if classify_dims is None:
            from dataset import CATEGORY_LIST, EMOTION_LIST
            classify_dims = [[64, len(EMOTION_LIST)], [64, len(CATEGORY_LIST)]]

        if use_classify:
            if use_multi_scale_emotion_encoder:
                self.multi_scale_emotion_encoder = MultiScaleEmotionEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_scales=3
                )
                emotion_input_dim = 768
                self.emotion_classify_net = ClassifyNet([emotion_input_dim] + classify_dims[0][1:])
            else:
                self.classify_compress_layer = nn.Conv1d(2*in_chans+1, 2*in_chans+1, kernel_size=patch_size*64, stride=patch_size*64)
                classify_dims[0][0] = (2*in_chans+1) * patch_size
                classify_dims[1][0] = (2*in_chans+1) * patch_size
                self.emotion_classify_net = ClassifyNet(classify_dims[0])

            if not use_multi_scale_emotion_encoder:
                self.category_classify_net = ClassifyNet(classify_dims[1])
            else:
                self.classify_compress_layer = nn.Conv1d(2*in_chans+1, 2*in_chans+1, kernel_size=patch_size*64, stride=patch_size*64)
                category_input_dim = (2*in_chans+1) * patch_size
                self.category_classify_net = ClassifyNet([category_input_dim] + classify_dims[1][1:])

            self.classify_nets = [self.emotion_classify_net, self.category_classify_net]
            self.classify_task_num = len(classify_dims)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)
        latent = self.forward_encoder(imgs)

        classify_preds = []
        if self.use_classify:
            if self.use_multi_scale_emotion_encoder:
                patch_tokens = latent[:, 1:, :]
                emotion_emb = self.multi_scale_emotion_encoder(patch_tokens)
                emotion_logits = self.emotion_classify_net(emotion_emb)
                emotion_pred = nn.Softmax(1)(emotion_logits)
                classify_preds.append(emotion_pred)
                classify_latent = self.classify_compress_layer(latent)
                classify_latent = torch.flatten(classify_latent, start_dim=1)
                category_logits = self.category_classify_net(classify_latent)
                category_pred = nn.Softmax(1)(category_logits)
                classify_preds.append(category_pred)
            else:
                classify_latent = self.classify_compress_layer(latent)
                classify_latent = torch.flatten(classify_latent, start_dim=1)
                for i, classify_model in enumerate(self.classify_nets):
                    logit = classify_model(classify_latent)
                    classify_pred = nn.Softmax(1)(logit)
                    classify_preds.append(classify_pred)

        latent = latent[:, 1:, :]
        return latent, classify_preds

    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)
        m, u = self.load_state_dict(state_dict, strict=False)
        return


class classify_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(64, 1, 1, stride=1)
        self.fc = nn.Linear(1024, 40)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class mapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(128, 1, 1, stride=1)
        self.fc = nn.Linear(1024, 768)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x
