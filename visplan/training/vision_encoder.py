"""Shared visual encoder for all trajectory prediction models.

Takes concatenated (start_image, goal_image, piece_mask) as a 7-channel
input and produces a global scene embedding via a ResNet-18 backbone.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class VisionEncoder(nn.Module):
    """ResNet-18 backbone adapted for 7-channel input → d-dim embedding.

    Input channels: start_image (3) + goal_image (3) + piece_mask (1) = 7.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = resnet18(weights=None)

        # Replace first conv: 3 → 7 input channels
        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # Initialise: copy pretrained weights for first 3 channels,
        # use Kaiming for the rest
        with torch.no_grad():
            self.conv1.weight[:, :3] = old_conv.weight
            nn.init.kaiming_normal_(self.conv1.weight[:, 3:], mode="fan_out",
                                    nonlinearity="relu")

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.proj = nn.Linear(512, embed_dim)

    def forward(self, start_image, goal_image, piece_mask):
        """
        Args:
            start_image: (B, 3, H, W)
            goal_image:  (B, 3, H, W)
            piece_mask:  (B, 1, H, W)

        Returns:
            (B, embed_dim) scene embedding.
        """
        x = torch.cat([start_image, goal_image, piece_mask], dim=1)  # (B, 7, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)       # (B, 512)
        return self.proj(x)    # (B, embed_dim)


class SpatialVisionEncoder(nn.Module):
    """ResNet-18 backbone that returns spatial feature maps for cross-attention.

    Returns both a global embedding and spatial features from layer3.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = resnet18(weights=None)

        self.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = backbone.conv1.weight
            nn.init.kaiming_normal_(self.conv1.weight[:, 3:], mode="fan_out",
                                    nonlinearity="relu")

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # output: (B, 256, H/16, W/16)

        self.spatial_proj = nn.Conv2d(256, embed_dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(256, embed_dim)

    def forward(self, start_image, goal_image, piece_mask):
        x = torch.cat([start_image, goal_image, piece_mask], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # (B, 256, h, w)

        spatial = self.spatial_proj(x)  # (B, embed_dim, h, w)
        B, C, h, w = spatial.shape
        spatial_tokens = spatial.flatten(2).permute(0, 2, 1)  # (B, h*w, embed_dim)

        glob = self.global_pool(x).flatten(1)    # (B, 256)
        glob = self.global_proj(glob)             # (B, embed_dim)

        return glob, spatial_tokens
