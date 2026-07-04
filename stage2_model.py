# stage2_model.py
# ç¬¬äºŒé˜¶æ®µä¸»æ¨¡åž‹å®šä¹‰ï¼šEEG + Sketch â†’ Image
# ä½¿ç”¨æ–°çš„CrossAttentionFusionèžåˆæ¨¡å— + EmotionAdapter æ³¨å…¥ + æƒ…æ„Ÿ/ç»“æž„æŸå¤±
#12/3 æ‹†åˆ†
#12/4 use_fusion å‚æ•°
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    T2IAdapter,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler
)

#from eeg_ldm_refining import get_clip_text_features, get_clip_image_features
from eeg_ldm_refining import (
    get_clip_text_features, 
    get_clip_image_features,
    init_sdxl_text_encoders,
    classify_sketch_with_clip,
    encode_sdxl_prompt,
    encode_sdxl_prompt_with_negative,
    build_prompt_from_category,
    CATEGORY_LIST,
    DEFAULT_NEGATIVE_PROMPT,
    _ensure_clip_components,
)

class EmotionAdapter(nn.Module):
    """
    å°†å…¨å±€ emotion_emb [B, D] æŠ•åˆ°æ¯ä¸ª down-block çš„é¢å¤–æ®‹å·®åˆ†æ”¯å½¢çŠ¶ï¼š
    - å¯¹äºŽ SDXL + T2I-Adapterï¼Œadapter è¿”å›žä¸€ä¸ª list[Tensor]ï¼Œæ¯å±‚å½¢çŠ¶ [B, C_i, H_i, W_i]
    - è¿™é‡Œæˆ‘ä»¬åªå­¦ä¹ "é€šé“åç½®"ï¼Œå†å¹¿æ’­åˆ°ç©ºé—´ï¼Œé¿å…å‚æ•°é‡è†¨èƒ€
    - ä¸Ž Adapter èžåˆï¼šres_i = (1-Î±_i)*adapter_i + Î±_i * (gate) * emo_i
      å…¶ä¸­ gate é»˜è®¤ä¸º 1ï¼ˆå¯æ‰©å±•ä¸ºå¯å­¦ä¹ é—¨æŽ§ï¼‰ï¼ŒÎ±_i âˆˆ [0,1] ä¸ºå¯å­¦ä¹ æ¯”ä¾‹ç³»æ•°ï¼ˆç»“æž„ vs æƒ…æ„Ÿï¼‰
    """
    def __init__(self, emotion_dim=768, adapter_channels=(320, 640, 1280, 1280)):
        super().__init__()
        self.adapter_channels = adapter_channels
        self.projs = nn.ModuleList()
        for c in adapter_channels:
            self.projs.append(
                nn.Sequential(
                    nn.Linear(emotion_dim, c),
                    nn.SiLU(),
                    nn.Linear(c, c)
                )
            )
        #self.alpha = nn.Parameter(torch.zeros(len(adapter_channels)))
        self.alpha = nn.Parameter(torch.ones(len(adapter_channels)) * -2.2)
        self.sigmoid = nn.Sigmoid()

        for m in self.projs:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.05)
                else:
                    nn.init.zeros_(p)

    def forward(self, emotion_emb, adapter_residuals):
        """
        emotion_emb: [B, D]
        adapter_residuals: list of [B, C_i, H_i, W_i] (å¯æœ‰ None)
        return: fused_residuals: list åŒå½¢çŠ¶
        """
        fused = []
        B = emotion_emb.size(0)
        for i, a in enumerate(adapter_residuals):
            if a is None:
                fused.append(None)
                continue
            C, H, W = a.shape[1], a.shape[2], a.shape[3]
            emo_vec = self.projs[i](emotion_emb)
            emo_map = emo_vec[:, :, None, None].to(dtype=a.dtype, device=a.device).expand(B, C, H, W)
            alpha_i = self.sigmoid(self.alpha[i])
            res = (1.0 - alpha_i) * a + alpha_i * emo_map
            fused.append(res)
        return fused
    def get_alphas(self):
        """è¿”å›žå½“å‰å„å±‚çš„èžåˆæ¯”ä¾‹ alpha (0=å…¨è‰å›¾, 1=å…¨è„‘ç”µ)"""
        return self.sigmoid(self.alpha).detach().cpu().numpy()

class EEGSketchToImageModel(nn.Module):
    """
    ç¬¬äºŒé˜¶æ®µä¸»æ¨¡åž‹
    - è‰å›¾è¯­ä¹‰ï¼šä½¿ç”¨ SketchSemanticEncoder çš„ CLS token
    - æƒ…æ„Ÿæ³¨å…¥ï¼šEmotionAdapter åœ¨ down-block ä½œä¸ºé¢å¤–åˆ†æ”¯æ³¨å…¥ï¼Œä¸Ž T2I-Adapter æ®‹å·®æŒ‰æ¯”ä¾‹èžåˆ
    - è®­ç»ƒæŸå¤±ï¼šåŸºç¡€æ‰©æ•£ MSE + æƒ…æ„Ÿè¯­ä¹‰æŸå¤± + ç»“æž„è¾¹ç¼˜æŸå¤±
    """
    def __init__(
        self,
        eeg_encoder,
        sketch_encoder,
        fusion_module,
        projector,
        adapter_path,
        sdxl_base_path,
        vae_path,
        device='cuda',
        use_fusion=True,
        emotion_dim=768,
        w_emotion=0.1,#æƒ…æ„ŸæŸå¤±æƒé‡
        w_edge=0.1#è¾¹ç¼˜æŸå¤±æƒé‡
    ):
        super().__init__()
        self.use_fusion = use_fusion
        self.device = device

        self.eeg_encoder = eeg_encoder
        self.sketch_encoder = sketch_encoder

        self.fusion_module = fusion_module
        self.projector = projector

        print(f"åŠ è½½T2I-Adapter...")
        self.adapter = T2IAdapter.from_pretrained(
            adapter_path, torch_dtype=torch.float16, local_files_only=True
        ).to(device)
        for param in self.adapter.parameters():
            param.requires_grad = False
        self.adapter.eval()

        print(f"åŠ è½½VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            vae_path, torch_dtype=torch.float16, local_files_only=True
        ).to(device)
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        print(f"åŠ è½½SDXL UNet...")
        self.unet = UNet2DConditionModel.from_pretrained(
            sdxl_base_path, subfolder="unet", torch_dtype=torch.float16, local_files_only=True
        ).to(device)
        for param in self.unet.parameters():
            param.requires_grad = False
        self.unet.eval()

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            sdxl_base_path, subfolder="scheduler", local_files_only=True
        )
        self.inference_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            sdxl_base_path, subfolder="scheduler", local_files_only=True
        )

        # Legacy text/category conditioning is loaded only if those helper methods are called.
        # The default stage2 path uses sketch CLS -> MLP directly for UNet cross-attention.
        self.sdxl_base_path = sdxl_base_path
        self.sdxl_tokenizer_1 = None
        self.sdxl_tokenizer_2 = None
        self.sdxl_text_encoder_1 = None
        self.sdxl_text_encoder_2 = None
        self.clip_model = None
        self.clip_processor = None

        self.emotion_adapter = EmotionAdapter(
            emotion_dim=emotion_dim,#768
            adapter_channels=(320, 640, 1280, 1280)
        ).to(device)

        self.w_emotion = w_emotion
        self.w_edge = w_edge

        print("âœ… æ¨¡åž‹åˆå§‹åŒ–å®Œæˆï¼ˆEmotionAdapter å·²å¯ç”¨ï¼ŒSketchè¯­ä¹‰=CLSï¼‰")
        self._print_trainable_params()

    

#è¾…åŠ©æ–¹æ³•
    def _print_trainable_params(self):
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters() if p.requires_grad) if self.fusion_module is not None else 0
        projector_params = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)
        emotion_params = sum(p.numel() for p in self.emotion_adapter.parameters() if p.requires_grad)
        total = fusion_params + projector_params + emotion_params
        print(f"ðŸ“Š å¯è®­ç»ƒå‚æ•°: Fusion={fusion_params:,}, Projector={projector_params:,}, EmotionAdapter={emotion_params:,}, Total={total:,}")

    def get_trainable_params(self):
        params = []
        if self.fusion_module is not None:
            params += list(self.fusion_module.parameters())
        params += list(self.projector.parameters())
        params += list(self.emotion_adapter.parameters())
        return params

    def get_emotion_emb(self, eeg_signal):
        """
        ç›´æŽ¥è°ƒç”¨MAEforEEGçš„forwardèŽ·å–emotion_emb
        
        Returns:
            emotion_emb: [B, 512] ä»ŽMultiScaleEmotionEncoderè¾“å‡º
        """
        with torch.no_grad():
            _, _, _, _, emotion_emb = self.eeg_encoder(
                eeg_signal,
                mask_ratio=0,
                classify_target_labels=[],
                mode='test'
            )
        return emotion_emb

    def get_sketch_embedding(self, sketch_images):
        """èŽ·å–è‰å›¾è¯­ä¹‰embeddingï¼ˆCLS tokenï¼‰"""
        with torch.no_grad():
            return self.sketch_encoder(sketch_images, return_cls=True)

    def get_adapter_features(self, sketch_images):
        sketch_resized = F.interpolate(
            sketch_images, size=(1024, 1024), mode='bilinear', align_corners=False
        ).to(dtype=torch.float16)
        with torch.no_grad():
            return self.adapter(sketch_resized)
        

    #1222æ–°å¢žæ–¹æ³•
    def _ensure_legacy_text_conditioners(self):
        if self.sdxl_tokenizer_1 is None:
            (
                self.sdxl_tokenizer_1,
                self.sdxl_tokenizer_2,
                self.sdxl_text_encoder_1,
                self.sdxl_text_encoder_2,
            ) = init_sdxl_text_encoders(self.sdxl_base_path, self.device, torch.float16)
        if self.clip_model is None:
            _ensure_clip_components(self.device)
            from eeg_ldm_refining import _clip_full_model, _clip_image_processor
            self.clip_model = _clip_full_model
            self.clip_processor = _clip_image_processor

    def get_clip_category_semantic(self, sketch_images):
        """
        Legacy helper: classify sketch with CLIP, then encode the category prompt with SDXL text encoders.
        
        Returns:
            encoder_hidden_states: [B, 77, 2048]
            text_embeds: [B, 1280]
            predicted_categories: List[str]
        """
        self._ensure_legacy_text_conditioners()
        # 1. CLIPåˆ†ç±»
        predicted_categories, _ = classify_sketch_with_clip(
            sketch_images, 
            self.clip_model, 
            self.clip_processor,
            CATEGORY_LIST,
            self.device
        )
        
        # 2. æž„å»ºprompt
        prompts = [build_prompt_from_category(cat) for cat in predicted_categories]
        
        # 3. SDXLæ–‡æœ¬ç¼–ç 
        encoder_hidden_states, text_embeds = encode_sdxl_prompt(
            prompts,
            self.sdxl_tokenizer_1,
            self.sdxl_tokenizer_2,
            self.sdxl_text_encoder_1,
            self.sdxl_text_encoder_2,
            self.device,
            torch.float16
        )
        
        return encoder_hidden_states, text_embeds, predicted_categories
    #1224 æ–°å¢žæ–¹æ³•ï¼šèŽ·å–å¸¦negativeçš„è¯­ä¹‰æ¡ä»¶
    def get_clip_category_semantic_with_negative(self, sketch_images, negative_prompt=None):
        """
        Legacy helper: classify sketch with CLIP and encode positive/negative SDXL text conditions.
        
        Returns:
            prompt_embeds: [B, 77, 2048]
            negative_prompt_embeds: [B, 77, 2048]
            pooled_prompt_embeds: [B, 1280]
            negative_pooled_prompt_embeds: [B, 1280]
            predicted_categories: List[str]
        """
        self._ensure_legacy_text_conditioners()
        B = sketch_images.size(0)
        
        # 1. CLIPåˆ†ç±»
        predicted_categories, _ = classify_sketch_with_clip(
            sketch_images, 
            self.clip_model, 
            self.clip_processor,
            CATEGORY_LIST,
            self.device
        )
        
        # 2. æž„å»ºprompt
        prompts = [build_prompt_from_category(cat) for cat in predicted_categories]
        
        # 3. å‡†å¤‡negative prompt
        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        negative_prompts = [negative_prompt] * B
        
        # 4. SDXLæ–‡æœ¬ç¼–ç ï¼ˆå¸¦negativeï¼‰
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = encode_sdxl_prompt_with_negative(
            prompts,
            negative_prompts,
            self.sdxl_tokenizer_1,
            self.sdxl_tokenizer_2,
            self.sdxl_text_encoder_1,
            self.sdxl_text_encoder_2,
            self.device,
            torch.float16
        )
        
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            predicted_categories
        )

    def get_time_ids(self, batch_size):
        return torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]] * batch_size,
            dtype=torch.float16, device=self.device
        )

    @staticmethod
    def _sobel_edges(x):
        """
        x: [B,3,H,W] in [-1,1] or [0,1]
        return: [B,1,H,W] å½’ä¸€åŒ–è¾¹ç¼˜å¹…å€¼
        """
        if x.min() < 0:
            x = (x + 1) / 2
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        kx = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], device=gray.device, dtype=gray.dtype).unsqueeze(0)
        ky = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], device=gray.device, dtype=gray.dtype).unsqueeze(0)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        mag = mag / (mag.amax(dim=[2,3], keepdim=True) + 1e-6)
        return mag

    def forward(self, eeg_signal, sketch_images, target_images, emotion_text_prompts=None, edge_loss_type="l1"):
        """Training forward pass."""
        B = eeg_signal.size(0)

        emotion_emb = self.get_emotion_emb(eeg_signal)
        sketch_emb = self.get_sketch_embedding(sketch_images)
        adapter_state = self.get_adapter_features(sketch_images)

        if self.use_fusion:
            fused_features = self.fusion_module(emotion_emb, sketch_emb, isFused=True)
            encoder_hidden_states, text_embeds = self.projector(fused_features)
        else:
            # Sketch CLS MLP path: Fs -> two-layer MLP -> SDXL cross-attention condition.
            encoder_hidden_states, text_embeds = self.projector(sketch_emb)

        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)
        text_embeds = text_embeds.to(dtype=torch.float16)
        time_ids = self.get_time_ids(B)

        target_resized = F.interpolate(
            target_images, size=(1024, 1024), mode='bilinear', align_corners=False
        ).to(dtype=torch.float16)

        with torch.no_grad():
            latents = self.vae.encode(target_resized).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        adapter_state_fp16 = [s.to(dtype=torch.float16) for s in adapter_state]
        fused_down_residuals = self.emotion_adapter(emotion_emb.to(self.device), adapter_state_fp16)

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            down_block_additional_residuals=fused_down_residuals,
        ).sample

        loss_diffusion = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        loss_total = loss_diffusion

        with torch.no_grad():
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(noisy_latents.device, dtype=noisy_latents.dtype)
            at = alphas_cumprod[timesteps].view(B, 1, 1, 1)
            sqrt_at = at.sqrt()
            sqrt_one_minus_at = (1 - at).sqrt()
            x0_latents = (noisy_latents - sqrt_one_minus_at * model_pred) / (sqrt_at + 1e-6)
            x0_latents = x0_latents / self.vae.config.scaling_factor
            recon_images = self.vae.decode(x0_latents).sample

        loss_emotion = torch.tensor(0.0, device=self.device)
        if self.w_emotion > 0 and emotion_text_prompts is not None:
            img_feat = get_clip_image_features(recon_images, device=self.device)
            txt_feat = get_clip_text_features(emotion_text_prompts, device=self.device)
            cos_sim = (img_feat * txt_feat).sum(dim=-1)
            loss_emotion = (1 - cos_sim).mean()
            loss_total = loss_total + self.w_emotion * loss_emotion

        loss_edge = torch.tensor(0.0, device=self.device)
        if self.w_edge > 0:
            edges_recon = self._sobel_edges(recon_images)
            sketch_resized = F.interpolate(
                sketch_images, size=recon_images.shape[-2:], mode='bilinear', align_corners=False
            )
            edges_sketch = self._sobel_edges(sketch_resized)
            if edge_loss_type == "l2":
                loss_edge = F.mse_loss(edges_recon, edges_sketch)
            else:
                loss_edge = F.l1_loss(edges_recon, edges_sketch)
            loss_total = loss_total + self.w_edge * loss_edge

        print(
            f"[Stage2] loss_total: {loss_total.item():.4f} = "
            f"diffusion: {loss_diffusion.item():.4f} + "
            f"emotion: {(self.w_emotion * loss_emotion).item():.4f} + "
            f"edge: {(self.w_edge * loss_edge).item():.4f}"
        )
        return {
            'total': loss_total,
            'diffusion': loss_diffusion,
            'emotion': loss_emotion,
            'edge': loss_edge,
            'emotion_weighted': self.w_emotion * loss_emotion,
            'edge_weighted': self.w_edge * loss_edge,
        }

    @torch.no_grad()
    def generate(self, eeg_signal, sketch_images, num_steps=50, guidance_scale=7.5,
                 negative_prompt=None, seed=None):
        """Generate images with sketch CLS MLP conditioning when use_fusion=False."""
        B = eeg_signal.size(0)
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        emotion_emb = self.get_emotion_emb(eeg_signal)
        sketch_emb = self.get_sketch_embedding(sketch_images)
        adapter_state = self.get_adapter_features(sketch_images)

        if self.use_fusion:
            fused_features = self.fusion_module(emotion_emb, sketch_emb, isFused=True)
            encoder_hidden_states, text_embeds = self.projector(fused_features)
        else:
            encoder_hidden_states, text_embeds = self.projector(sketch_emb)

        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)
        text_embeds = text_embeds.to(dtype=torch.float16)
        time_ids = self.get_time_ids(B)

        adapter_state_fp16 = [s.to(dtype=torch.float16) for s in adapter_state]
        fused_down_residuals = self.emotion_adapter(emotion_emb.to(self.device), adapter_state_fp16)

        latents = torch.randn(
            (B, 4, 128, 128),
            device=self.device,
            dtype=torch.float16,
            generator=generator,
        )
        self.inference_scheduler.set_timesteps(num_steps, device=self.device)
        latents = latents * self.inference_scheduler.init_noise_sigma

        for t in self.inference_scheduler.timesteps:
            latent_model_input = self.inference_scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                down_block_additional_residuals=fused_down_residuals,
            ).sample
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def save_checkpoint(self, path, epoch, optimizer, scheduler, loss):
        torch.save({
            'epoch': epoch,
            'fusion_module': self.fusion_module.state_dict() if self.fusion_module is not None else None,
            'projector': self.projector.state_dict(),
            'emotion_adapter': self.emotion_adapter.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'w_emotion': self.w_emotion,
            'w_edge': self.w_edge
        }, path)
        print(f"âœ… Checkpointä¿å­˜åˆ°: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        if self.fusion_module is not None and ckpt.get('fusion_module') is not None:
            self.fusion_module.load_state_dict(ckpt['fusion_module'])
        self.projector.load_state_dict(ckpt['projector'])
        if 'emotion_adapter' in ckpt:
            self.emotion_adapter.load_state_dict(ckpt['emotion_adapter'])
        if 'w_emotion' in ckpt:
            self.w_emotion = ckpt['w_emotion']
        if 'w_edge' in ckpt:
            self.w_edge = ckpt['w_edge']
        print(f"âœ… CheckpointåŠ è½½è‡ª: {path}")
        return ckpt


