# eeg_ldm_refining.py
# ç¬¬äºŒé˜¶æ®µæ¨¡å—å®šä¹‰ï¼šEEG Encoderã€Sketch Encoderã€èžåˆæ¨¡å—ã€æŠ•å½±æ¨¡å—
#
####12/5å‡Œæ™¨ç‰ˆæœ¬ï¼Œæ¢äº†æ–°çš„æŠ•å½±æ¨¡å—
# 2. SemanticProjector ä¿®æ”¹
# 3. SemanticProjector è¾“å‡ºä¹˜ä»¥ 3.0ï¼Œå¯¹é½çœŸå®ž Text Encoder çš„æ•°å€¼èŒƒå›´
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor, CLIPTokenizer
# ============================================================
# å…¨å±€å•ä¾‹
# ============================================================
_sketch_encoder = None
_eeg_encoder = None
_clip_full_model = None
_clip_image_processor = None
_clip_tokenizer = None

_sdxl_tokenizer_1 = None
_sdxl_tokenizer_2 = None
_sdxl_text_encoder_1 = None
_sdxl_text_encoder_2 = None

# ç±»åˆ«åˆ—è¡¨ï¼ˆä¸Ždataset.pyä¿æŒä¸€è‡´ï¼‰
CATEGORY_LIST = [
    'airplane', 'ant', 'bear', 'bell', 'butterfly', 'camel', 'car',
    'castle', 'chair', 'couch', 'crab', 'duck', 'elephant', 'eyeglasses',
    'fish', 'flower', 'giraffe', 'guitar', 'horse', 'hot_air_balloon',
    'kangaroo', 'knife', 'lion', 'monkey', 'mouse', 'mushroom', 'owl',
    'panda', 'parrot', 'penguin', 'piano', 'pineapple', 'rabbit',
    'rhinoceros', 'rifle', 'sailboat', 'snail', 'snake', 'spider',
    'strawberry', 'teddy_bear', 'tiger', 'tree', 'umbrella', 'violin',
    'whale', 'windmill', 'zebra'
]

# é»˜è®¤negative prompt
DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, blurry, monochrome, grayscale, deformed, bad anatomy, ugly, distorted, pixelated, noisy, watermark, text, disfigured"

# ============================================================
# ä¸´æ—¶å‡½æ•°ï¼šå°†latentè½¬æ¢ä¸ºemotion_embç»´åº¦
# ============================================================

# ============================================================
# Sketchè¯­ä¹‰ç¼–ç å™¨ï¼ˆè¿”å›žCLSï¼‰
# ============================================================
class SketchSemanticEncoder(nn.Module):
    def __init__(self, clip_model_path):
        super().__init__()
        print(f"æ­£åœ¨åŠ è½½CLIPæ¨¡åž‹: {clip_model_path}")
        full_clip_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        self.vision_model = full_clip_model.vision_model
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_path, local_files_only=True)
        # å†»ç»“
        for p in self.vision_model.parameters():
            p.requires_grad = False
        self.vision_model.eval()
        print("âœ… è‰å›¾è¯­ä¹‰ç¼–ç å™¨åˆå§‹åŒ–å®Œæˆï¼ˆè¾“å‡ºCLSï¼‰")
    @torch.no_grad()
    def forward(self, sketch_images, return_cls=True):
        device = sketch_images.device
        pil_images = []
        for i in range(sketch_images.size(0)):
            img = sketch_images[i]
            if img.max() <= 1.0:
                img = (img * 255).clamp(0, 255).byte()
            img_np = img.permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)

        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        # hidden_states: list of layer outputs [B, seq, C]ï¼Œå–æœ€åŽä¸€å±‚
        last_hidden = outputs.hidden_states[-1]  # [B, 257, 1024] for ViT-L
        cls_token = last_hidden[:, 0, :]         # [B, 1024]
        if return_cls:
            return cls_token
        else:
            # ä¸Žæ—§ç‰ˆå…¼å®¹ï¼šè¿”å›ž pooler_output ç­‰ä»·
            return outputs.pooler_output  # [B, 1024]
def init_sketch_encoder(config, device=None):
    """åˆå§‹åŒ–è‰å›¾è¯­ä¹‰ç¼–ç å™¨ï¼ˆå•ä¾‹æ¨¡å¼ï¼‰"""
    global _sketch_encoder
    if _sketch_encoder is None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/pretrains/clip-vit-large-patch14"
        _sketch_encoder = SketchSemanticEncoder(clip_model_path).to(device)
        print("âœ… è‰å›¾è¯­ä¹‰ç¼–ç å™¨å·²åŠ è½½å¹¶ç¼“å­˜ï¼ˆCLSï¼‰")
    return _sketch_encoder
# ============================================================
# EEG Encoderåˆå§‹åŒ–
# ============================================================
def init_eeg_encoder(checkpoint_path, device):
    """åˆå§‹åŒ–EEG Encoderï¼ˆå•ä¾‹æ¨¡å¼ï¼‰"""
    global _eeg_encoder
    if _eeg_encoder is None:
        print(f"æ­£åœ¨åŠ è½½EEG Encoder: {checkpoint_path}")
        from sc_mbm.mae_for_eeg import MAEforEEG
        _eeg_encoder = MAEforEEG(
            time_len=512, patch_size=4, embed_dim=1024, in_chans=64,
            depth=24, num_heads=16, decoder_embed_dim=512, decoder_depth=8,
            decoder_num_heads=16, mlp_ratio=1., use_classify=True,
            use_multi_scale_emotion_encoder=True, classify_dims=[[64, 8], [64, 48]]
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        _eeg_encoder.load_state_dict(checkpoint['model'], strict=False)
        _eeg_encoder = _eeg_encoder.to(device)
        for p in _eeg_encoder.parameters():
            p.requires_grad = False
        _eeg_encoder.eval()
        print("âœ… EEG EncoderåŠ è½½å®Œæˆï¼ˆfrozenï¼‰")
    return _eeg_encoder
# ============================================================
# CLIPï¼ˆæ–‡æœ¬/å›¾åƒå¡”ï¼‰ç”¨äºŽæƒ…æ„Ÿè¯­ä¹‰æŸå¤±ï¼ˆå•ä¾‹ï¼‰
# ============================================================
def _ensure_clip_components(device=None):
    global _clip_full_model, _clip_image_processor, _clip_tokenizer
    # æ–°å¢žï¼šè‡ªåŠ¨æ£€æµ‹è®¾å¤‡ï¼ˆå¦‚æžœæœªä¼ å…¥deviceï¼‰
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _clip_full_model is None:
        from transformers import CLIPProcessor  # æ–°å¢žå¯¼å…¥
        
        clip_model_path = "/home/dell/dreamdiffusion_project/DreamDiffusion/pretrains/clip-vit-large-patch14"
        _clip_full_model = CLIPModel.from_pretrained(clip_model_path, local_files_only=True)
        _clip_full_model = _clip_full_model.to(device)
        _clip_full_model.eval()
        for p in _clip_full_model.parameters():
            p.requires_grad = False
        # ä¿®æ”¹ï¼šä½¿ç”¨CLIPProcessoræ›¿ä»£CLIPImageProcessor
        _clip_image_processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
        _clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=True)
        print("âœ… CLIP ç»„ä»¶ï¼ˆæ–‡æœ¬/å›¾åƒï¼‰å·²å°±ç»ªï¼ˆfrozenï¼‰")
@torch.no_grad()

#1222 æ–°å¢žSDXLæ–‡æœ¬ç¼–ç å™¨åˆå§‹åŒ–å‡½æ•°
def init_sdxl_text_encoders(sdxl_base_path, device, dtype=torch.float16):
    """åˆå§‹åŒ–SDXLçš„åŒæ–‡æœ¬ç¼–ç å™¨ï¼ˆå•ä¾‹æ¨¡å¼ï¼‰"""
    global _sdxl_tokenizer_1, _sdxl_tokenizer_2, _sdxl_text_encoder_1, _sdxl_text_encoder_2
    
    if _sdxl_text_encoder_1 is None:
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        
        print(f"åŠ è½½SDXL Text Encoder 1 (CLIP ViT-L)...")
        _sdxl_tokenizer_1 = CLIPTokenizer.from_pretrained(
            sdxl_base_path, subfolder="tokenizer", local_files_only=True
        )
        _sdxl_text_encoder_1 = CLIPTextModel.from_pretrained(
            sdxl_base_path, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True
        ).to(device).eval()
        
        print(f"åŠ è½½SDXL Text Encoder 2 (CLIP ViT-bigG)...")
        _sdxl_tokenizer_2 = CLIPTokenizer.from_pretrained(
            sdxl_base_path, subfolder="tokenizer_2", local_files_only=True
        )
        _sdxl_text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            sdxl_base_path, subfolder="text_encoder_2", torch_dtype=dtype, local_files_only=True
        ).to(device).eval()
        
        # å†»ç»“å‚æ•°
        for p in _sdxl_text_encoder_1.parameters():
            p.requires_grad = False
        for p in _sdxl_text_encoder_2.parameters():
            p.requires_grad = False
            
        print("âœ… SDXLåŒæ–‡æœ¬ç¼–ç å™¨åŠ è½½å®Œæˆï¼ˆfrozenï¼‰")
    
    return _sdxl_tokenizer_1, _sdxl_tokenizer_2, _sdxl_text_encoder_1, _sdxl_text_encoder_2

#1222æ–°å¢žè‰å›¾CLIPåˆ†ç±»å‡½æ•°
def classify_sketch_with_clip(sketch_images, clip_model, clip_processor, category_list, device):
    """
    ä½¿ç”¨CLIPå¯¹è‰å›¾è¿›è¡Œåˆ†ç±»ï¼Œè¿”å›žé¢„æµ‹çš„ç±»åˆ«åç§°åˆ—è¡¨
    
    Args:
        sketch_images: [B, 3, H, W] tensorï¼Œå€¼èŒƒå›´[0,1]æˆ–[-1,1]
        clip_model: CLIPæ¨¡åž‹
        clip_processor: CLIPå¤„ç†å™¨
        category_list: ç±»åˆ«åç§°åˆ—è¡¨
        device: è®¾å¤‡
    
    Returns:
        predicted_categories: List[str]ï¼Œé•¿åº¦ä¸ºBçš„é¢„æµ‹ç±»åˆ«åç§°åˆ—è¡¨
    """
    B = sketch_images.size(0)
    
    # è½¬æ¢ä¸ºPILå›¾åƒ
    pil_images = []
    for i in range(B):
        img = sketch_images[i]
        # å¤„ç†å€¼èŒƒå›´
        if img.min() < 0:
            img = (img + 1) / 2
        img = img.clamp(0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        pil_images.append(Image.fromarray(img_np))
    
    # å‡†å¤‡å¤šä¸ªpromptæ¨¡æ¿
    prompt_templates = [
        "a sketch of {}",
        "a drawing of {}",
        "a simple drawing of {}",
        "an outline drawing of {}",
        "a black and white sketch of {}"
    ]
    
    all_probs = []
    
    for template in prompt_templates:
        text_prompts = [template.format(category) for category in category_list]
        
        # å¯¹æ¯å¼ å›¾åƒåˆ†åˆ«å¤„ç†
        batch_probs = []
        for pil_img in pil_images:
            inputs = clip_processor(
                text=text_prompts, 
                images=pil_img, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits = outputs.logits_per_image  # [1, num_categories]
                probs = logits.softmax(dim=1)
                batch_probs.append(probs)
        
        # [B, num_categories]
        batch_probs = torch.cat(batch_probs, dim=0)
        all_probs.append(batch_probs)
    
    # å¹³å‡æ‰€æœ‰æ¨¡æ¿çš„é¢„æµ‹ç»“æžœ
    final_probs = torch.stack(all_probs).mean(dim=0)  # [B, num_categories]
    
    # èŽ·å–é¢„æµ‹ç±»åˆ«
    pred_indices = torch.argmax(final_probs, dim=1)  # [B]
    predicted_categories = [category_list[idx.item()] for idx in pred_indices]
    
    return predicted_categories, final_probs

#1222æ–°å¢žSDXLæ–‡æœ¬ç¼–ç å‡½æ•°
def encode_sdxl_prompt(
    prompts,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    device,
    dtype=torch.float16,
):
    """
    ä½¿ç”¨SDXLçš„åŒæ–‡æœ¬ç¼–ç å™¨ç¼–ç prompt
    
    Args:
        prompts: List[str]ï¼Œé•¿åº¦ä¸ºBçš„promptåˆ—è¡¨
        
    Returns:
        encoder_hidden_states: [B, 77, 2048]
        pooled_prompt_embeds: [B, 1280]
    """
    # Text Encoder 1
    text_input_1 = tokenizer_1(
        prompts, 
        padding="max_length", 
        max_length=77, 
        truncation=True, 
        return_tensors="pt"
    )
    
    # Text Encoder 2
    text_input_2 = tokenizer_2(
        prompts, 
        padding="max_length", 
        max_length=77, 
        truncation=True, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        # Encoder 1
        out_1 = text_encoder_1(
            text_input_1.input_ids.to(device),
            attention_mask=text_input_1.attention_mask.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_embeds_1 = out_1.hidden_states[-2]  # [B, 77, 768]
        
        # Encoder 2
        out_2 = text_encoder_2(
            text_input_2.input_ids.to(device),
            attention_mask=text_input_2.attention_mask.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_embeds_2 = out_2.hidden_states[-2]  # [B, 77, 1280]
        pooled_prompt_embeds = out_2.text_embeds   # [B, 1280]
    
    # æ‹¼æŽ¥å¾—åˆ° [B, 77, 2048]
    encoder_hidden_states = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    
    return (
        encoder_hidden_states.to(dtype=dtype),
        pooled_prompt_embeds.to(dtype=dtype),
    )

#1224æ–°å¢žnegative promptå‡½æ•°
def encode_sdxl_prompt_with_negative(
    prompts,
    negative_prompts,
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    device,
    dtype=torch.float16,
):
    """
    ä½¿ç”¨SDXLçš„åŒæ–‡æœ¬ç¼–ç å™¨ç¼–ç promptå’Œnegative prompt
    
    Args:
        prompts: List[str]ï¼Œé•¿åº¦ä¸ºBçš„promptåˆ—è¡¨
        negative_prompts: List[str]ï¼Œé•¿åº¦ä¸ºBçš„negative promptåˆ—è¡¨
        
    Returns:
        prompt_embeds: [B, 77, 2048]
        negative_prompt_embeds: [B, 77, 2048]
        pooled_prompt_embeds: [B, 1280]
        negative_pooled_prompt_embeds: [B, 1280]
    """
    # ---- ç¼–ç  positive prompt ----
    text_input_1 = tokenizer_1(
        prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    text_input_2 = tokenizer_2(
        prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        out_1 = text_encoder_1(
            text_input_1.input_ids.to(device),
            attention_mask=text_input_1.attention_mask.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_embeds_1 = out_1.hidden_states[-2]  # [B, 77, 768]

        out_2 = text_encoder_2(
            text_input_2.input_ids.to(device),
            attention_mask=text_input_2.attention_mask.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        prompt_embeds_2 = out_2.hidden_states[-2]  # [B, 77, 1280]
        pooled_prompt_embeds = out_2.text_embeds   # [B, 1280]

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)  # [B, 77, 2048]

    # ---- ç¼–ç  negative prompt ----
    neg_input_1 = tokenizer_1(
        negative_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    neg_input_2 = tokenizer_2(
        negative_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        neg_out_1 = text_encoder_1(
            neg_input_1.input_ids.to(device),
            attention_mask=neg_input_1.attention_mask.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        neg_embeds_1 = neg_out_1.hidden_states[-2]  # [B, 77, 768]

        neg_out_2 = text_encoder_2(
            neg_input_2.input_ids.to(device),
            attention_mask=neg_input_2.attention_mask.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        neg_embeds_2 = neg_out_2.hidden_states[-2]  # [B, 77, 1280]
        neg_pooled_embeds = neg_out_2.text_embeds   # [B, 1280]

    negative_prompt_embeds = torch.cat([neg_embeds_1, neg_embeds_2], dim=-1)  # [B, 77, 2048]

    return (
        prompt_embeds.to(dtype=dtype),
        negative_prompt_embeds.to(dtype=dtype),
        pooled_prompt_embeds.to(dtype=dtype),
        neg_pooled_embeds.to(dtype=dtype),
    )

#122 æ–°å¢žåŸºäºŽç±»åˆ«ç”Ÿæˆpromptçš„å‡½æ•°
def build_prompt_from_category(category_name):
    """æ ¹æ®ç±»åˆ«åç§°æž„å»ºç”Ÿæˆprompt"""
    return f"a high quality, detailed, realistic image of a {category_name}, natural colors, sharp focus"

def get_clip_text_features(prompts, device):
    _ensure_clip_components(device)
    inputs = _clip_tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    out = _clip_full_model.get_text_features(**inputs)  # [B, 768 or 1024]
    out = F.normalize(out, dim=-1)
    return out
@torch.no_grad()
def get_clip_image_features(images, device):
    """
    images: [B, 3, H, W] in [-1, 1] æˆ– [0,1]ï¼Œä¼šè‡ªåŠ¨è½¬ PIL å†èµ° processor
    """
    _ensure_clip_components(device)
    B = images.size(0)
    pil_images = []
    for i in range(B):
        img = images[i]
        img = ((img + 1) / 2).clamp(0, 1) if img.min() < 0 else img.clamp(0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        pil_images.append(Image.fromarray(img_np))
    inputs = _clip_image_processor(images=pil_images, return_tensors="pt").to(device)
    out = _clip_full_model.get_image_features(**inputs)
    out = F.normalize(out, dim=-1)
    return out
# ============================================================
# èžåˆæ¨¡å—ï¼šCross-Attention Fusionï¼ˆä¿æŒä¸å˜ï¼‰
# ============================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, emotion_dim=512, sketch_dim=1024, hidden_dim=1024, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emotion_proj = nn.Linear(emotion_dim, hidden_dim)
        self.sketch_proj = nn.Linear(sketch_dim, hidden_dim)
        self.cross_attn_e2s = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.cross_attn_s2e = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(hidden_dim))
        print(f"âœ… CrossAttentionFusionåˆå§‹åŒ–å®Œæˆ (emotion_dim={emotion_dim}, sketch_dim={sketch_dim}, hidden_dim={hidden_dim})")
    def forward(self, emotion_emb, sketch_emb, isFused=True):
        if isFused:
            emotion = self.emotion_proj(emotion_emb)
            sketch = self.sketch_proj(sketch_emb)
            emotion = emotion.unsqueeze(1)
            sketch = sketch.unsqueeze(1)
            emotion_attended, _ = self.cross_attn_e2s(query=emotion, key=sketch, value=sketch)
            sketch_attended, _ = self.cross_attn_s2e(query=sketch, key=emotion, value=emotion)
            emotion_attended = emotion_attended.squeeze(1)
            sketch_attended = sketch_attended.squeeze(1)
            gate = self.gate(torch.cat([emotion_attended, sketch_attended], dim=-1))
            fused = gate * emotion_attended + (1 - gate) * sketch_attended
            fused = self.fusion_mlp(fused)
            return fused
        else:
            sketch = self.sketch_proj(sketch_emb)
            fused = self.fusion_mlp(sketch)
            return fused
# ============================================================
# æŠ•å½±æ¨¡å—ï¼ˆä¿æŒä¸å˜ï¼‰
# ============================================================
class SketchSemanticProjector(nn.Module):
    """Project sketch CLS features into SDXL conditioning space.

    The sketch semantic feature is mapped by a two-layer MLP with SiLU
    activations and LayerNorm. The projected token is used directly as
    encoder_hidden_states for the SDXL UNet cross-attention, while a
    parallel pooled projection supplies SDXL's added text_embeds.
    """
    def __init__(self, input_dim=1024, hidden_dim=2048, pooled_dim=1280, output_scale=3.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_scale = output_scale
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pooled_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pooled_dim),
            nn.LayerNorm(pooled_dim),
        )
        total_params = sum(p.numel() for p in self.parameters())
        print(f"SketchSemanticProjector initialized (input_dim={input_dim}, hidden_dim={hidden_dim}, pooled_dim={pooled_dim})")
        print(f"   parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    def forward(self, sketch_cls):
        token = self.token_mlp(sketch_cls)
        encoder_hidden_states = token.unsqueeze(1) * self.output_scale
        text_embeds = self.pooled_mlp(sketch_cls)
        return encoder_hidden_states, text_embeds
class SemanticProjector(nn.Module):
    def __init__(self, input_dim=1024, seq_len=77, hidden_dim=2048, pooled_dim=1280):
        super().__init__()
        #self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        #self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        #self.seq_transform = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.to_pooled = nn.Sequential(nn.Linear(input_dim, pooled_dim), nn.LayerNorm(pooled_dim))
        self.output_scale = 3.0
        total_params = sum(p.numel() for p in self.parameters())
        print(f"âœ… SemanticProjectoråˆå§‹åŒ–å®Œæˆ (input_dim={input_dim}, seq_len={seq_len})")
        print(f"   å‚æ•°é‡: {total_params:,} (~{total_params/1e6:.1f}M)")
    def forward(self, fused):
        B = fused.size(0)
        proj = self.proj(fused)# [B, hidden_dim]
        #seq = proj.unsqueeze(1).expand(B, self.seq_len, -1)
        #seq = seq + self.pos_embedding
        encoder_hidden_states = proj.unsqueeze(1) * self.output_scale  # [B, 1, hidden_dim]
        text_embeds = self.to_pooled(fused)# [B, pooled_dim]
        return encoder_hidden_states, text_embeds

