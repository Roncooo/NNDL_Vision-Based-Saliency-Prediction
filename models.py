"""
Model architectures for saliency prediction.

Contract (frozen): every model maps an RGB image (B,3,H,W) to a saliency
map (B,1,H,W) with values in [0,1]. New architectures are added with
@register_model and selected by name in config.py  train.py never changes.
"""

import torch
import torch.nn as nn
from torchvision import models as tvm

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def _up_block(in_ch, out_ch):
    """Deconv block: doubles spatial resolution, then BN + ReLU.

    kernel=4, stride=2, padding=1 gives an exact x2 upsample; choosing the
    kernel divisible by the stride avoids the classic checkerboard artifacts
    of transposed convolutions (Odena et al., 2016).
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BaselineCNN(nn.Module):
    """Plain CNN encoder decoder: the stage-1 baseline of the project plan.

    Encoder  VGG16 conv layers, pretrained on ImageNet, as in Wang & Shen
    (Deep Visual Attention Prediction, TIP 2018), who build their attention
    net on the same backbone: SALICON (~10k images) is far too small to learn
    good low-level filters from scratch, so we transfer them. The FC
    classifier head is dropped (dense prediction needs a spatial map, not a
    class vector) and, following the paper, the last max-pool (pool5) is
    removed so features stay at H/16 instead of H/32  for per-pixel
    prediction, spatial resolution is worth more than extra downsampling.
    The encoder is NOT frozen: it is fine-tuned end-to-end with the decoder
    (as in the paper), which works fine with a small lr (e.g. 1e-4).

    Decoder  4 deconv blocks, each doubling resolution (H/16 -> H) while
    halving channels (512 -> 32); reducing channels while upsampling keeps
    memory/compute reasonable (same scheme as the paper's decoder). A final
    1x1 conv maps the 32 feature maps to a single channel and a sigmoid
    squashes it into [0,1] to satisfy the output contract.

    Note: H and W must be multiples of 16 (4 pooling stages, 4 upsamples).
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tvm.vgg16(weights=weights)
        # vgg.features = conv1_1 ... conv5_3 + pool5; [:-1] drops only pool5
        self.encoder = vgg.features[:-1]          # (B,3,H,W) -> (B,512,H/16,W/16)
        self.decoder = nn.Sequential(
            _up_block(512, 256),                  # H/16 -> H/8
            _up_block(256, 128),                  # H/8  -> H/4
            _up_block(128, 64),                   # H/4  -> H/2
            _up_block(64, 32),                    # H/2  -> H
            nn.Conv2d(32, 1, kernel_size=1),      # 32 feature maps -> 1 saliency channel
            nn.Sigmoid(),                         # per-pixel values in [0,1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@register_model("BaselineCNN")
def build_baseline_cnn(config):
    return BaselineCNN(pretrained=getattr(config, "pretrained", True))


class VGGMultiScaleEncoder(nn.Module):
    """Shared VGG16 trunk tapped at three depths (Wang & Shen, TIP 2018).

    The paper selects M=3 feature maps from conv3_3, conv4_3 and conv5_3
    ("considering further more layers does not contribute to performance
    improvement, but brings extra computation burden") and learns all scales
    "within a single network", i.e. one shared trunk instead of three
    separate encoders. Slicing torchvision's vgg16.features:
      [:16]   conv1_1 .. conv3_3+ReLU -> (B,256,H/4, W/4)
      [16:23] pool3  .. conv4_3+ReLU -> (B,512,H/8, W/8)
      [23:30] pool4  .. conv5_3+ReLU -> (B,512,H/16,W/16)  (pool5 dropped, as in BaselineCNN)
    Pretrained on ImageNet and fine-tuned end-to-end, same transfer-learning
    argument as the baseline.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        features = tvm.vgg16(weights=weights).features
        self.stage3 = features[:16]
        self.stage4 = features[16:23]
        self.stage5 = features[23:30]

    def forward(self, x):
        f3 = self.stage3(x)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return f3, f4, f5


class DecoderBranch(nn.Module):
    """Per-scale decoder: deconv blocks up to full resolution -> 1-channel logits.

    Mirrors the paper's per-stream decoders: 2/3/4 deconvolution layers for
    the conv3_3/conv4_3/conv5_3 streams, "each deconvolution layer doubles
    the spatial size" -- trainable upsampling, "more favored than ... a fixed
    bilinear interpolation kernel". `channels` is the full schedule, e.g.
    [512,256,128,64,32] = 4 up-blocks. No sigmoid here: branches emit logits
    and a single sigmoid is applied once, after fusion.
    """

    def __init__(self, channels):
        super().__init__()
        self.up = nn.Sequential(
            *[_up_block(c_in, c_out) for c_in, c_out in zip(channels[:-1], channels[1:])]
        )
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, x):
        return self.head(self.up(x))


class SkipDecoderBranch(nn.Module):
    """Per-scale decoder WITH U-Net skip connections -> 1-channel logits.

    Same role as DecoderBranch (upsample one encoder tap back to full
    resolution), but after an up-block reaches a resolution where an encoder
    tap exists, that tap is concatenated onto the feature map before the next
    up-block. This is the classic U-Net skip (Ronneberger et al., MICCAI
    2015): pooling in the encoder discards fine spatial detail that a deep
    feature map can no longer recover on its own, so the higher-resolution
    encoder activations are fed straight into the decoder to sharpen
    localization. It is a strict addition on top of stage-2's multi-scale
    fusion: the fusion combines *whole* per-scale predictions, whereas these
    skips reinject encoder detail *inside* a single branch's upsampling path.

    Interface (kept explicit on purpose so the wiring is auditable):
      out_chs : output channels of each up-block, e.g. [256,128,64,32].
                Deliberately identical to the matching DecoderBranch so the
                ONLY new variable vs. MultiScaleCNN is the concatenated skips.
      skip_chs: channels concatenated AFTER each up-block (same length as
                out_chs); 0 where the reached resolution has no encoder tap.
    forward(x, skips): `skips` is a list aligned to the up-blocks, holding the
    encoder tensor to concatenate after each up-block, or None where skip_ch
    is 0. Concatenation only ever happens at matching resolution (guaranteed
    by how MultiScaleSkipCNN wires the taps).
    """

    def __init__(self, in_ch, out_chs, skip_chs):
        super().__init__()
        assert len(out_chs) == len(skip_chs), "one skip slot per up-block"
        self.ups = nn.ModuleList()
        c = in_ch
        for out_ch, skip_ch in zip(out_chs, skip_chs):
            self.ups.append(_up_block(c, out_ch))   # x2 upsample to out_ch
            c = out_ch + skip_ch                    # next block sees the concat
        self.head = nn.Conv2d(c, 1, kernel_size=1)  # c = last out (+ last skip, =0 here)

    def forward(self, x, skips):
        assert len(skips) == len(self.ups), "one skip (or None) per up-block"
        for up, skip in zip(self.ups, skips):
            x = up(x)
            if skip is not None:
                # resolutions are matched by construction (see MultiScaleSkipCNN)
                x = torch.cat([x, skip], dim=1)
        return self.head(x)


class FusionModule(nn.Module):
    """Wang & Shen's "attention fusion" layer.

    A 1x1 convolution over the stacked per-scale maps "simultaneously learns
    the fusion weight during training" (F = sum_m w_m * S_m); one sigmoid
    then maps the fused result into [0,1]. Same pattern as HED's side-output
    fusion (Xie & Tu, ICCV 2015), including the initialization to a uniform
    average of the streams so training starts from a sensible fusion.
    """

    def __init__(self, num_maps):
        super().__init__()
        self.fuse = nn.Conv2d(num_maps, 1, kernel_size=1)
        nn.init.constant_(self.fuse.weight, 1.0 / num_maps)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, maps):
        return torch.sigmoid(self.fuse(torch.cat(maps, dim=1)))


class MultiScaleCNN(nn.Module):
    """Multi-scale saliency network: stage-2 model of the project plan
    (Wang & Shen, "Deep Visual Attention Prediction", IEEE TIP 2018).

    Saliency is driven both by low-level contrast (Itti et al., PAMI 1998)
    and by high-level content such as faces/people (Judd et al., ICCV 2009),
    which live at different depths of a CNN. So instead of decoding only the
    deepest feature map (BaselineCNN), three per-scale decoders each predict
    a saliency map from conv3_3 / conv4_3 / conv5_3 (fine/local to coarse/
    semantic) and a learned 1x1-conv fusion merges them. Combining shallow
    and deep layers is the classic dense-prediction recipe (FCN, Long et al.
    CVPR 2015; Hypercolumns, Hariharan et al. CVPR 2015).

    The dec5 branch is identical to the baseline decoder, so any metric gain
    is attributable to the extra scales + fusion, not to a different decoder.
    Deviation from the paper: no deep supervision (per-stream losses would
    break the one-output/one-loss training contract); only the fused map is
    supervised.

    Input constraint as baseline: H and W must be multiples of 16.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = VGGMultiScaleEncoder(pretrained=pretrained)
        self.dec3 = DecoderBranch([256, 64, 32])            # H/4  -> H (2 up-blocks)
        self.dec4 = DecoderBranch([512, 128, 64, 32])       # H/8  -> H (3 up-blocks)
        self.dec5 = DecoderBranch([512, 256, 128, 64, 32])  # H/16 -> H (4, = baseline decoder)
        self.fusion = FusionModule(num_maps=3)

    def forward(self, x):
        f3, f4, f5 = self.encoder(x)
        return self.fusion([self.dec3(f3), self.dec4(f4), self.dec5(f5)])


@register_model("MultiScaleCNN")
def build_multiscale_cnn(config):
    return MultiScaleCNN(pretrained=getattr(config, "pretrained", True))


class MultiScaleSkipCNN(nn.Module):
    """Multi-scale + skip-connection saliency network: stage-3 of the plan.

    Identical to MultiScaleCNN (same shared VGG encoder, same three taps,
    same fusion layer) EXCEPT the three decoder branches are replaced by
    SkipDecoderBranch, so each branch concatenates the higher-resolution
    encoder taps into its upsampling path (U-Net skips, Ronneberger et al.
    MICCAI 2015). Encoder and fusion are unchanged, so this is a clean
    ablation isolating the effect of adding skip connections on top of the
    stage-2 multi-scale model.

    Which taps skip into which branch (only ever at matching resolution,
    with tap resolutions f3@H/4, f4@H/8, f5@H/16):
      dec5 (from f5@H/16): ->H/8 concat f4, ->H/4 concat f3, ->H/2, ->H
      dec4 (from f4@H/8):  ->H/4 concat f3, ->H/2, ->H
      dec3 (from f3@H/4):  ->H/2, ->H            (no shallower tap -> no skips)
    Up-block output channels match MultiScaleCNN's branches exactly; only the
    *input* channels grow where a skip is concatenated (that extra capacity
    is the point of the stage, and its param cost is reported in the notes).

    Input constraint as before: H and W must be multiples of 16.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = VGGMultiScaleEncoder(pretrained=pretrained)
        # out_chs mirror MultiScaleCNN; skip_chs = encoder tap channels
        # concatenated after the up-block that reaches the tap's resolution.
        self.dec3 = SkipDecoderBranch(256, [64, 32],           [0, 0])            # no skips
        self.dec4 = SkipDecoderBranch(512, [128, 64, 32],      [256, 0, 0])       # +f3 @ H/4
        self.dec5 = SkipDecoderBranch(512, [256, 128, 64, 32], [512, 256, 0, 0])  # +f4 @ H/8, +f3 @ H/4
        self.fusion = FusionModule(num_maps=3)

    def forward(self, x):
        f3, f4, f5 = self.encoder(x)
        s3 = self.dec3(f3, [None, None])
        s4 = self.dec4(f4, [f3, None, None])
        s5 = self.dec5(f5, [f4, f3, None, None])
        return self.fusion([s3, s4, s5])


@register_model("MultiScaleSkipCNN")
def build_multiscale_skip_cnn(config):
    return MultiScaleSkipCNN(pretrained=getattr(config, "pretrained", True))


def build_model(config):
    """Factory function called by train.py"""
    name = config.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)
