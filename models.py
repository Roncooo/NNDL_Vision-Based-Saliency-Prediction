"""
Model architectures for saliency prediction.

Contract (frozen): every model maps an RGB image (B,3,H,W) to a saliency
map (B,1,H,W) with values in [0,1]. New architectures are added with
@register_model and selected by name in config.py — train.py never changes.
"""

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
    """Plain CNN encoder–decoder: the stage-1 baseline of the project plan.

    Encoder — VGG16 conv layers, pretrained on ImageNet, as in Wang & Shen
    (Deep Visual Attention Prediction, TIP 2018), who build their attention
    net on the same backbone: SALICON (~10k images) is far too small to learn
    good low-level filters from scratch, so we transfer them. The FC
    classifier head is dropped (dense prediction needs a spatial map, not a
    class vector) and, following the paper, the last max-pool (pool5) is
    removed so features stay at H/16 instead of H/32 — for per-pixel
    prediction, spatial resolution is worth more than extra downsampling.
    The encoder is NOT frozen: it is fine-tuned end-to-end with the decoder
    (as in the paper), which works fine with a small lr (e.g. 1e-4).

    Decoder — 4 deconv blocks, each doubling resolution (H/16 -> H) while
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


def build_model(config):
    """Factory function called by train.py"""
    name = config.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in registry. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)
