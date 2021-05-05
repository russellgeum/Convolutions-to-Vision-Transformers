import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange



class LinearPatchEmbbeding(nn.Module):
    def __init__(self, in_channels: int, patch_height: int, patch_width: int, embbeding_dim: int):
        super(LinearPatchEmbbeding, self).__init__()
        self.rearrange = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = patch_height, s2 = patch_width)
        self.linear    = nn.Linear(patch_height * patch_width * in_channels, embbeding_dim)
    
    def forward(self, images):
        """
        Args:
            images: [B, C, H, W]
            
            H = h * patch_size
            W = w * patch_size
        return
            [B, h * w, patch_size * patch_size, C]
        """
        features = self.rearrange(images)
        features = self.linear(features)
        return features


class ConvPatchEmbbeding(nn.Module):
    def __init__(self, in_channels: int, patch_height: int, patch_width: int, embbeding_dim: int):
        super(ConvPatchEmbbeding, self).__init__()
        self.conv2d    = nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = embbeding_dim,
                            kernel_size = (patch_height, patch_width),
                            stride = (patch_height, patch_width))
        self.rearrange = Rearrange('b e h w -> b (h w) e')

    def forward(self, images):
        """
        Args:
            images: [B, C, H, W]
            
            H = h * patch_size
            W = w * patch_size
        return
            [B, h * w, patch_size * patch_size, C]
        """
        features = self.conv2d(images)
        features = self.rearrange(features)
        return features
