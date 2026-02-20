import os
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# CONFIG: relative LoRA path for GitHub
DEFAULT_LORA_PATH = "Florence2/Saved_Models/epoch_12"
BASE_MODEL_ID = "microsoft/Florence-2-base-ft"


class VisionEncoder(nn.Module):
    """
    Wraps the Florence-2 vision tower into a clean encoder.

    forward(pixel_values) -> (image_tokens, patch_feats)

    - image_tokens: [B, L_vis, D_proj]  visual tokens actually seen by the LM
    - patch_feats:  [B, H_p, W_p, C_last] last-stage vision features (for heatmaps)
    """

    def __init__(self, florence_model: AutoModelForCausalLM):
        super().__init__()
        self.vision_tower = florence_model.vision_tower

        # These come from Florence-2
        self.image_projection = florence_model.image_projection
        self.image_proj_norm = florence_model.image_proj_norm
        self.image_pos_embed = florence_model.image_pos_embed
        self.visual_temporal_embed = florence_model.visual_temporal_embed
        self.image_feature_source = florence_model.image_feature_source

    def forward(self, pixel_values: torch.Tensor):
        """
        pixel_values: [B, 3, H, W], preprocessed by the Florence processor.

        Returns:
            image_tokens: [B, L_vis, D_proj]
            patch_feats : [B, H_p, W_p, C_last]
        """
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W), got {pixel_values.shape}")

        device = pixel_values.device
        B, C, H, W = pixel_values.shape
        T = 1  # single frame

        # 1. Backbone features (unpooled)
        x = self.vision_tower.forward_features_unpool(pixel_values)  # [B*T, N, C_last]
        N = x.shape[1]
        side = int(N ** 0.5)
        assert side * side == N, "Expected square feature map."

        patch_feats = x.view(B * T, side, side, x.shape[-1])  # [B*T, H_p, W_p, C_last]

        # 2. Positional embedding
        if self.image_pos_embed is not None:
            x_grid = patch_feats  # [B*T, H_p, W_p, C_last]
            pos = self.image_pos_embed(x_grid)
            x_grid = x_grid + pos
            x = x_grid.view(B, T * side * side, x_grid.shape[-1])  # [B, T*N, C_last]
        else:
            x = x.view(B, T * N, x.shape[-1])

        # 3. Temporal embedding (even if T=1)
        if self.visual_temporal_embed is not None:
            x_view = x.view(B, T, -1, x.shape[-1])          # [B, T, N, C_last]
            temporal_input = x_view[:, :, 0]                # [B, T, C_last]
            temporal_embed = self.visual_temporal_embed(temporal_input)
            x_view = x_view + temporal_embed.view(B, T, 1, -1)
            x = x_view.view(B, T * side * side, x.shape[-1])

        # 4. Build feature dict
        x_feat = {}
        x_view = x.view(B, T, -1, x.shape[-1])  # [B, T, N, C_last]

        spatial_avg = x_view.mean(dim=2)        # [B, T, C_last]
        temporal_avg = x_view.mean(dim=1)       # [B, C_last]
        last_frame = x_view[:, -1]              # [B, N, C_last]

        x_feat["spatial_avg_pool"] = spatial_avg
        x_feat["temporal_avg_pool"] = temporal_avg
        x_feat["last_frame"] = last_frame

        # Concatenate selected feature sources along sequence dimension
        feats = []
        for src in self.image_feature_source:
            if src not in x_feat:
                raise ValueError(f"Unknown image_feature_source: {src}")
            feats.append(x_feat[src])
        x_cat = torch.cat(feats, dim=1)        # [B, L_vis, C_last]

        # 5. Project into joint visual-language space
        x_proj = x_cat @ self.image_projection  # [B, L_vis, D_proj]
        x_proj = self.image_proj_norm(x_proj)

        return x_proj, patch_feats


def load_florence_vision(
    lora_path: str = DEFAULT_LORA_PATH,
    device: str | None = None,
):
    """
    Loads Florence-2 base + LoRA and returns:
        model, processor, vision_encoder, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading Florence-2 model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    # Load LoRA adapter if it exists (relative path for GitHub)
    if lora_path and os.path.exists(lora_path):
        print(f"[INFO] Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("[INFO] LoRA merged into main model.")
    else:
        print("[WARN] LoRA path not found. Using base Florence-2.")

    # Freeze vision tower
    if hasattr(model, "vision_tower"):
        for p in model.vision_tower.parameters():
            p.requires_grad = False
        print("[INFO] Vision tower frozen.")
    else:
        raise AttributeError("model has no attribute 'vision_tower'")

    model.eval()

    vision_encoder = VisionEncoder(model).to(device)
    vision_encoder.eval()

    return model, processor, vision_encoder, device

