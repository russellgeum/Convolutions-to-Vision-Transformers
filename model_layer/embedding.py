import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange



class LinePatchEmbbeding(nn.Module):
    def __init__(self, in_channels: int, patch_size: tuple, embbeding_dim: int):
        super(LinePatchEmbbeding, self).__init__()
        self.rearrange = Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = patch_size[0], s2 = patch_size[1])
        self.linear    = nn.Linear(patch_size[0] * patch_size[1] * in_channels, embbeding_dim)
    
    def forward(self, images):
        """
        Args:
            images: [B, C, H, W]
            
        1. 패치 사이즈로 쪼개서 height는 패치의 수, width는 패치의 길이들에 채널을 곱해서 옆으로 펼침, [B, h*w, patch_size*patch_size*C]
        2. [B, h*w, patch_size*patch_size*C] 를 [patch_size*patch_size*C, embbeding_dim]으로 리니어 매핑하여 임베딩
        return
            features: [B, h*w, embedding_dim]
            즉 패치 하나하나가 embedding_dim의 차원을 가진 벡터로 임베딩되었다.
        """
        features = self.rearrange(images)
        features = self.linear(features)
        return features



class ConvPatchEmbbeding(nn.Module):
    def __init__(self, in_channels: int, patch_size: tuple, embbeding_dim: int):
        super(ConvPatchEmbbeding, self).__init__()
        self.conv2d    = nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = embbeding_dim,
                            kernel_size = (patch_size[0], patch_size[1]),
                            stride = (patch_size[0], patch_size[1]))
        self.rearrange = Rearrange('b c h w -> b (h w) c')

    def forward(self, images):
        """
        Args:
            images: [B, C, H, W]
            H = h * patch_size
            W = w * patch_size
        
        1. 컨볼루션으로 커널 사이즈는 패치 크기로, 스트라이드도 패치 크기로 해서 임베딩 먼저 계산
        2. 차원이 [배치, 임베딩 채널, 패치 세로 수, 패치 가로 수] (패치의 수 == 패치 세로 수 x 패치 가로 수)
        3. [배치, embedding_dim, patch_height, patch_width]를 reshape하여 [배치, patch_height, patch_width, embedding_dim]으로 리턴
        return
            features: [B, h*w, embedding_dim]
        """
        features = self.conv2d(images)
        features = self.rearrange(features)
        return features
