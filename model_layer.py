import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from einops.layers.torch import Rearrange
from model_module import *



class SelfAttention_ConvolutionTransfomer(nn.Module):
    """
    셀프 어텐션 컨볼루션 트랜스포머
    """
    def __init__(self, size, in_channels, depth = 1, heads = 8, dim_head = 8, dim_mlp = 64):
        super(SelfAttention_ConvolutionTransfomer, self).__init__()
        """
        Args:
            size:  [B C H W]
            depth: int
            heads: int
            dim_head: int
            in_channels: int
            dlm_mlp: int
        """
        self.in_channels   = in_channels
        self.forward_shape = Rearrange('b c h w -> b (h w) c', h = size[0], w = size[1])
        self.LayerNorm     = nn.LayerNorm(in_channels)
        self.module        = nn.ModuleList([])
        
        # depth는 트랜스포머 레이어의 수
        for _ in range(depth):
            self.module.append(
                nn.ModuleList([
                    SelfPreNorm(in_channels, 
                            ConvSelfAttention(
                                size = size, 
                                heads = heads, 
                                dim_head = dim_head, 
                                in_channels = in_channels)),
                    SelfPreNorm(in_channels, FeedFoward(in_channels, dim_mlp))]))

        self.inverse_shape = Rearrange('b (h w) c -> b c h w', h = size[0], w = size[1])

    def forward(self, features):
        features = self.forward_shape(features)
        features = self.LayerNorm(features)

        for attention, feedforward in self.module:
            features = attention(features) + features
            features = feedforward(features) + features

        features = self.inverse_shape(features)
        return features



class CrossAttention_ConvolutionTransformer(nn.Module):
    """
    크로스 어텐션 컨볼루션 트랜스포머
    """
    def __init__(self, size1, size2, in_channels, heads, dim_head, dim_mlp):
        super(CrossAttention_ConvolutionTransformer, self).__init__()
        self.in_channels         = in_channels
        self.self_forward_shape  = Rearrange('b c h w -> b (h w) c', h = size1[0], w = size1[1])
        self.cross_forward_shape = Rearrange('b c h w -> b (h w) c', h = size2[0], w = size2[1])
        self.LayerNorm           = nn.LayerNorm(in_channels)
        
        self.Layer1 = CrossPreNorm(in_channels, 
                              ConvCrossAttention(
                                    size1 = size1,
                                    size2 = size2,
                                    in_channels = in_channels,
                                    heads = heads, 
                                    dim_head = dim_head))
        self.Layer2 = SelfPreNorm(in_channels, FeedFoward(in_channels, dim_mlp))
                    
        self.inverse_shape = Rearrange('b (h w) c -> b c h w', h = size2[0], w = size2[1])

    def forward(self, self_features, cross_features):
        """
        Args:
            self_features:  [B, C, H, W]
            cross_features: [B, C', H', W']
        """
        # (B C H W) -> (B HW C), 모양을 바꾸는 코드
        self_features = self.self_forward_shape(self_features)
        self_features = self.LayerNorm(self_features)
        
        # (B C H W) -> (B HW C), 모양을 바꾸는 코드
        cross_features = self.cross_forward_shape(cross_features)
        cross_features = self.LayerNorm(cross_features)
        
        # self_features: (B HW C), cross_features: (B HW C)
        # 이 안에서 self_features 크기가 cross_features로 복원이 됨
        features = self.Layer1(self_features, cross_features) + cross_features
        features = self.Layer2(features)
        features = self.inverse_shape(features)
        return features