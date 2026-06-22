"""
model.py — Embed2Heights dual-branch model.

Architecture (matches architecture SVG exactly):
  Branch A — High-Resolution (AlphaEarth 64ch + Tessera 128ch → 192ch):
    Pretrained ConvNeXt-Base backbone via timm → 5 multi-scale feature maps
    (256×256, 128×128, 64×64, 32×32, 16×16), all projected to D=256 channels.

  Branch B — Patch Token (TerraMind S1+S2 concat 1536ch + THOR S1+S2 concat 1536ch → 3072ch):
    Linear projection → 4-layer Transformer Encoder → (B, D, 16, 16)

  Fusion — Cross-Attention at 16×16:
    HR features = Query, Token features = Key/Value
    Pre-norm + residual + FFN

  Decoder — SegFormer-style:
    All 5 HR scales + fused tokens → 1×1 proj → bilinear upsample to 256×256
    → concatenate → Conv3×3 fuse → (B, D, 256, 256)

  Heads:
    Segmentation : Conv → sigmoid  → (B, 3, 256, 256)  building/veg/water ∈ [0,1]
    Height       : Conv → softplus → (B, 1, 256, 256)  nDSM ≥ 0 (log1p space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Shared building blocks
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1  = ConvBnReLU(ch, ch)
        self.c2  = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.c2(self.c1(x)) + x)


# ══════════════════════════════════════════════════════════════════════════════
# Branch A — High-Resolution Encoder
# ══════════════════════════════════════════════════════════════════════════════

class HREncoderConvNeXt(nn.Module):
    """
    Pretrained ConvNeXt-Base backbone (timm).
    Stem is replaced to accept 192 input channels.
    Pretrained weights are preserved via mean-initialised channel expansion.

    ConvNeXt-Base stage output channels: [128, 256, 512, 1024]
    For a 256×256 input, spatial sizes after each stage:
      stage0 →  64×64   (stride 4)
      stage1 →  32×32   (stride 8)
      stage2 →  16×16   (stride 16)
      stage3 →   8×8    (stride 32, unused)

    We produce 5 decoder-ready feature maps:
      F1: 256×256×D   (upsample ×4 from stage0)
      F2: 128×128×D   (upsample ×2 from stage0)
      F3:  64× 64×D   (stage0 projected)
      F4:  32× 32×D   (stage1 projected)
      F5:  16× 16×D   (stage2 projected) ← cross-attention input
    """
    _STAGE_CHS = [128, 256, 512, 1024]

    def __init__(self, in_ch: int = 192, D: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2),   # only need first 3 stages
        )

        # ── Replace stem Conv2d(3→128) with Conv2d(192→128) ──────────────
        orig_stem = self.backbone.stem
        orig_conv = orig_stem[0]
        new_conv  = nn.Conv2d(
            in_ch, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=(orig_conv.bias is not None),
        )
        with torch.no_grad():
            # Average pretrained 3-ch weights → 1-ch, tile to in_ch, rescale
            w_mean   = orig_conv.weight.mean(dim=1, keepdim=True)   # (128,1,4,4)
            repeats  = in_ch // 3
            leftover = in_ch  % 3
            w_new    = w_mean.repeat(1, repeats * 3 + max(leftover, 0), 1, 1)[:, :in_ch]
            w_new    = w_new * (3.0 / in_ch)
            new_conv.weight.copy_(w_new)
            if orig_conv.bias is not None:
                new_conv.bias.copy_(orig_conv.bias)
        stem_list    = list(orig_stem)
        stem_list[0] = new_conv
        self.backbone.stem = nn.Sequential(*stem_list)

        # ── 1×1 projections: each stage output → D channels ──────────────
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, D, 1, bias=False), nn.BatchNorm2d(D), nn.ReLU(inplace=True))
            for c in self._STAGE_CHS[:3]
        ])

    def forward(self, alpha: torch.Tensor, tessera: torch.Tensor):
        x          = torch.cat([alpha, tessera], dim=1)   # (B, 192, 256, 256)
        s0, s1, s2 = self.backbone(x)
        # s0: (B, 128,  64, 64)
        # s1: (B, 256,  32, 32)
        # s2: (B, 512,  16, 16)
        p0, p1, p2 = self.proj[0](s0), self.proj[1](s1), self.proj[2](s2)

        f1 = F.interpolate(p0, size=(256, 256), mode="bilinear", align_corners=False)
        f2 = F.interpolate(p0, size=(128, 128), mode="bilinear", align_corners=False)
        f3 = p0    # 64×64
        f4 = p1    # 32×32
        f5 = p2    # 16×16
        return f1, f2, f3, f4, f5


class HREncoderFallback(nn.Module):
    """
    Pure-PyTorch fallback when timm is not available.
    ConvNeXt-style (patch stem stride-4, ResBlocks) — same output contract as above.
    """
    def __init__(self, in_ch: int = 192, D: int = 256, **_):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(in_ch, D, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(D), nn.ReLU(inplace=True),
        )                                                # → (B, D, 64, 64)
        self.stage0 = nn.Sequential(ResBlock(D), ResBlock(D))
        self.down1  = ConvBnReLU(D, D, k=3, s=2, p=1)  # → 32×32
        self.stage1 = nn.Sequential(ResBlock(D), ResBlock(D))
        self.down2  = ConvBnReLU(D, D, k=3, s=2, p=1)  # → 16×16
        self.stage2 = nn.Sequential(ResBlock(D), ResBlock(D))

    def forward(self, alpha: torch.Tensor, tessera: torch.Tensor):
        x  = torch.cat([alpha, tessera], dim=1)
        p0 = self.stage0(self.stem(x))
        p1 = self.stage1(self.down1(p0))
        p2 = self.stage2(self.down2(p1))

        f1 = F.interpolate(p0, size=(256, 256), mode="bilinear", align_corners=False)
        f2 = F.interpolate(p0, size=(128, 128), mode="bilinear", align_corners=False)
        f3, f4, f5 = p0, p1, p2
        return f1, f2, f3, f4, f5


HREncoder = HREncoderConvNeXt if TIMM_AVAILABLE else HREncoderFallback


# ══════════════════════════════════════════════════════════════════════════════
# Branch B — Patch Token Encoder
# ══════════════════════════════════════════════════════════════════════════════

class TokenEncoder(nn.Module):
    """
    Input : tokens (B, 1536, 16, 16) + thor (B, 1536, 16, 16)
    Output: (B, D, 16, 16)

    Steps:
      1. Concatenate along channel → (B, 3072, 16, 16)
      2. Flatten spatial → sequence (B, 256, 3072)
      3. Linear proj → (B, 256, token_dim)
      4. Transformer Encoder (n_layers, n_heads)
      5. Linear out_proj → (B, 256, D)
      6. Reshape back → (B, D, 16, 16)
    """
    def __init__(self, in_ch: int = 3072, token_dim: int = 512,
                 n_heads: int = 8, n_layers: int = 4, D: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_ch, token_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=n_heads,
            dim_feedforward=token_dim * 2,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj    = nn.Linear(token_dim, D)

    def forward(self, tokens: torch.Tensor, thor: torch.Tensor):
        x        = torch.cat([tokens, thor], dim=1)    # (B, 3072, 16, 16)
        B, C, H, W = x.shape
        x        = x.flatten(2).permute(0, 2, 1)       # (B, 256, 3072)
        x        = self.proj(x)                         # (B, 256, token_dim)
        x        = self.transformer(x)                  # (B, 256, token_dim)
        x        = self.out_proj(x)                     # (B, 256, D)
        return x.permute(0, 2, 1).reshape(B, -1, H, W) # (B, D, 16, 16)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Attention Fusion
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Both inputs at 16×16×D.
    HR features → Query.  Token features → Key, Value.
    Pre-norm + residual + FFN (standard transformer block pattern).
    """
    def __init__(self, D: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q   = nn.LayerNorm(D)
        self.norm_kv  = nn.LayerNorm(D)
        self.attn     = nn.MultiheadAttention(D, n_heads, dropout=dropout, batch_first=True)
        self.ffn      = nn.Sequential(
            nn.Linear(D, D * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(D * 2, D)
        )
        self.norm_ffn = nn.LayerNorm(D)

    def forward(self, hr_feat: torch.Tensor, tok_feat: torch.Tensor):
        B, D, H, W = hr_feat.shape
        q   = hr_feat.flatten(2).permute(0, 2, 1)      # (B, HW, D)
        kv  = tok_feat.flatten(2).permute(0, 2, 1)     # (B, HW, D)
        # Cross-attention with pre-norm
        attn_out, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        q = q + attn_out                                # residual
        q = q + self.ffn(self.norm_ffn(q))              # FFN residual
        return q.permute(0, 2, 1).reshape(B, D, H, W)  # (B, D, 16, 16)


# ══════════════════════════════════════════════════════════════════════════════
# SegFormer-style Decoder
# ══════════════════════════════════════════════════════════════════════════════

class SegFormerDecoder(nn.Module):
    """
    Inputs:  f1(256×256), f2(128×128), f3(64×64), f4(32×32), fused(16×16) — all D channels
    Process: 1×1 proj on each → bilinear upsample all to 256×256 → cat → Conv3×3 fuse
    Output:  (B, D, 256, 256)
    """
    def __init__(self, D: int = 256, n_inputs: int = 5):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(D, D, 1) for _ in range(n_inputs)])
        self.fuse = nn.Sequential(
            nn.Conv2d(D * n_inputs, D, 3, 1, 1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            ResBlock(D),
        )

    def forward(self, f1, f2, f3, f4, fused):
        target = f1.shape[-2:]
        feats  = [f1, f2, f3, f4, fused]
        ups    = []
        for i, f in enumerate(feats):
            p = self.proj[i](f)
            if p.shape[-2:] != target:
                p = F.interpolate(p, size=target, mode="bilinear", align_corners=False)
            ups.append(p)
        return self.fuse(torch.cat(ups, dim=1))   # (B, D, 256, 256)


# ══════════════════════════════════════════════════════════════════════════════
# Full Model
# ══════════════════════════════════════════════════════════════════════════════

class Embed2HeightsModel(nn.Module):
    """
    Full dual-branch multi-task model.

    Inputs  (all CHW tensors):
      alpha   : (B, 64,   256, 256)
      tessera : (B, 128,  256, 256)
      tokens  : (B, 1536, 16,  16)   TerraMind S1+S2
      thor    : (B, 1536, 16,  16)   THOR S1+S2

    Outputs:
      seg    : (B, 3, 256, 256)  sigmoid  — building/veg/water fraction ∈ [0,1]
      height : (B, 1, 256, 256)  softplus — nDSM in log1p space (invert with expm1)
    """

    def __init__(self, D: int = 256, token_dim: int = 512,
                 n_heads: int = 8, n_tx_layers: int = 4,
                 pretrained: bool = True):
        super().__init__()
        self.hr_encoder    = HREncoder(in_ch=192, D=D, pretrained=pretrained)
        self.token_encoder = TokenEncoder(in_ch=3072, token_dim=token_dim,
                                          n_heads=n_heads, n_layers=n_tx_layers, D=D)
        self.fusion        = CrossAttentionFusion(D=D, n_heads=n_heads)
        self.decoder       = SegFormerDecoder(D=D, n_inputs=5)

        self.seg_head    = nn.Sequential(ConvBnReLU(D, D // 2), nn.Conv2d(D // 2, 3, 1))
        self.height_head = nn.Sequential(ConvBnReLU(D, D // 2), nn.Conv2d(D // 2, 1, 1))

    def forward(self, alpha, tessera, tokens, thor):
        f1, f2, f3, f4, f5 = self.hr_encoder(alpha, tessera)
        tok_feat            = self.token_encoder(tokens, thor)
        fused               = self.fusion(f5, tok_feat)
        decoded             = self.decoder(f1, f2, f3, f4, fused)

        seg    = torch.sigmoid(self.seg_head(decoded))
        height = F.softplus(self.height_head(decoded))
        return seg, height

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{n/1e6:.1f}M"


def build_model(cfg: dict, pretrained: bool = True) -> Embed2HeightsModel:
    return Embed2HeightsModel(
        D           = cfg["D"],
        token_dim   = cfg["token_dim"],
        n_heads     = cfg["n_heads"],
        n_tx_layers = cfg["n_tx_layers"],
        pretrained  = pretrained,
    )


if __name__ == "__main__":
    import sys
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"timm   : {'available' if TIMM_AVAILABLE else 'NOT FOUND — using fallback encoder'}")

    cfg = dict(D=256, token_dim=512, n_heads=8, n_tx_layers=4)
    model = build_model(cfg, pretrained=False).to(device)
    print(f"Params : {model.param_count()}")

    with torch.no_grad():
        seg, ht = model(
            torch.zeros(1, 64,   256, 256, device=device),
            torch.zeros(1, 128,  256, 256, device=device),
            torch.zeros(1, 1536, 16,  16,  device=device),
            torch.zeros(1, 1536, 16,  16,  device=device),
        )
    print(f"seg    : {tuple(seg.shape)}  range [{seg.min():.3f}, {seg.max():.3f}]")
    print(f"height : {tuple(ht.shape)}   range [{ht.min():.3f}, {ht.max():.3f}]")
    print("model.py self-test passed.")
