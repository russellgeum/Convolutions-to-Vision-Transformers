import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
from einops.layers.torch import Rearrange

from model_module import *



class SelfAttention_ConvolutionTransfomer(nn.Module):
    """
    Self-Attention을 사용하는 컨볼루션 트랜스포머
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
    Cross-Attention을 사용하는 컨볼루션 트랜스포머
    """
    def __init__(self, size1, size2, in_channels, heads = 8, dim_head = 8, dim_mlp = 64):
        super(CrossAttention_ConvolutionTransformer, self).__init__()
        """
        Args:
            size: turple or list
            patch_size: turple or list
            in_channels: int
            depth: int, Depth of MHSA
            heads:    int
            dim_head: int
            heads * dim_head -> hidden dimension of Q, K, V
            dim_mlp : MLP dimension of attention
        """
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



class TokenSelfAttention_ConvolutionTransformer(nn.Module):
    def __init__(self, size, patch_size, in_channels, depth = 1, heads = 8, dim_head = 8, dim_mlp = 64):
        super(TokenSelfAttention_ConvolutionTransformer, self).__init__()
        """
        Args:
            size: turple or list
            patch_size: turple or list
            in_channels: int
            depth: int, Depth of MHSA
            heads:    int
            dim_head: int
            heads * dim_head -> hidden dimension of Q, K, V
            dim_mlp : MLP dimension of attention
        """
        self.patch_height = patch_size[0]
        self.patch_width  = patch_size[1]
        self.patch_units  = (int(size[0] / patch_size[0]), int(size[1] / patch_size[1]))

        self.spliter    = TensorSplit(size, patch_size)
        self.stacker    = TensorStack(size, patch_size)

        module_list       = []
        for index in range(self.patch_units[0] * self.patch_units[1]):
            layer = SelfAttention_ConvolutionTransfomer(
                            (self.patch_height, self.patch_width), 
                            in_channels, 
                            depth, 
                            heads, 
                            dim_head, 
                            dim_mlp)
            module_list.append(layer)
        self.module_list  = nn.ModuleList(module_list)
    
    def forward(self, features):
        splited_features = self.spliter(features)
        
        feature_list = []
        for module, patch in zip(self.module_list, splited_features):
            out = module(patch)
            feature_list.append(out)

        outputs = self.stacker(feature_list)
        return outputs



class TokenCrossAttention_ConvolutionTransformer(nn.Module):
    """
    피처맵이나 이미지가 너무 큰 경우 토큰화를 해서 각 패치에 대하여 크로스 어텐션을 수행
    이 함수의 경우 두 피처맵을 토큰화 할때 같은 사이즈의 패치로 나누더라도, 패치의 갯수가 다르면 계산이 불가능
    따라서 피처맵의 사이즈와 패치 사이즈가 무조건 동일해야 함

    이후 대응해야 할 부분
    패치의 사이즈가 달라도 패치의 갯수가 같으면 계산이 가능하므로,
    이는 이후 크로스 어텐션 내부에서 interpolate하는 방법으로 보완 가능
    """
    def __init__(self, size1, size2, patch_size, in_channels, heads = 8, dim_head = 8, dim_mlp = 64):
        super(TokenCrossAttention_ConvolutionTransformer, self).__init__()
        """
        Args:
            size: turple or list
            patch_size: turple or list
            in_channels: int
            depth: int, Depth of MHSA
            heads:    int
            dim_head: int
            heads * dim_head -> hidden dimension of Q, K, V
            dim_mlp : MLP dimension of attention
        """
        self.patch_height = patch_size[0]
        self.patch_width  = patch_size[1]
        self.patch_units  = (int(size1[0] / patch_size[0]), int(size1[1] / patch_size[1]))

        self.self_spliter  = TensorSplit(size1, patch_size)
        self.cross_spliter = TensorSplit(size2, patch_size)
        self.stacker       = TensorStack(size1, patch_size)

        module_list       = []
        for index in range(self.patch_units[0] * self.patch_units[1]):
            layer = CrossAttention_ConvolutionTransformer(
                            (self.patch_height, self.patch_width),
                            (self.patch_height, self.patch_width),
                            in_channels,
                            heads, 
                            dim_head, 
                            dim_mlp)
            module_list.append(layer)
        self.module_list  = nn.ModuleList(module_list)

    def forward(self, self_features, cross_features):
        splited_self  = self.self_spliter(self_features)
        splited_cross = self.cross_spliter(cross_features)

        feature_list  = []
        for module, self_patch, cross_patch in zip(self.module_list, splited_self, splited_cross):
            out = module(self_patch, cross_patch)
            feature_list.append(out)

        outputs = self.stacker(feature_list)
        return outputs