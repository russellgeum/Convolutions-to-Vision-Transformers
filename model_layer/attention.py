import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange



"""
README
class LayerNorm
class SeparableConv2D
class FeedForward
class ConvolutionAttnetion (CvT)
"""
class LayerNorm(nn.Module):
    def __init__(self, in_channels):
        super(LayerNorm, self).__init__()
        """
        Attention, FFN 이전에 LayerNorm을 계산
        """
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        out = self.norm(x)
        return out



class SeparableConv2D(nn.Module):
    def __init__(self, 
        in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(SeparableConv2D, self).__init__()
        """
        Separatble Convoltuion
        1. 채널 수는 유지하고 3x3 컨볼루션 계산
        2. BatchNormalization2d
        3. 채널 수를 늘리고 1x1 컨볼루션 계산

        Args:
            in_channels: int
            out_channels: int
            kernel_size: turple
        """
        self.depth_wise = nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = in_channels)
        self.batchnorm  = nn.BatchNorm2d(in_channels)
        self.pointwise  = nn.Conv2d(
            in_channels =in_channels,
            out_channels = out_channels,
            kernel_size = (1, 1))
    
    def forward(self, inputs):
        out = self.depth_wise(inputs)
        out = self.batchnorm(out)
        out = self.pointwise(out)
        return out



class FeedForward(nn.Module):
    def __init__(self, in_channels ,dim_mlp):
        super(FeedForward, self).__init__()
        """
        FeedForward Network
        """
        self.net = nn.Sequential(
            nn.Linear(in_channels, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, in_channels))
    
    def forward(self, inputs):
        out = self.net(inputs)
        return out



class ConvolutionAttention(nn.Module):
    def __init__(self, 
        size, in_channels, kernel_size = (3, 3), 
        heads = 8, dim_head = 8, q_stride = 1, k_stride = 1, v_stride = 1):
        super(ConvolutionAttention, self).__init__()
        """
        Args
            size:               셀프 피처맵의 사이즈
            heads:              헤드의 갯수
            dim_head:           헤드 하나의 차원 수
            in_channels:        어텐션 모듈에 입력받을 피처맵의 채널 ([B, C, H, W]에서 C에 해당)
            kernel_size:        (kernel_height, kernel_width),
            q_stride, k_stride, v_stride: Q, K, V에서 작동할 컨볼루션의 스트라이드
        """
        self.size         = size
        self.in_channels  = in_channels
        self.kernel_size  = kernel_size
        self.heads        = heads
        self.dim_head     = dim_head
        self.out_channels = dim_head * heads
 
        # heads == 1이고 dim_head == in_channels이면 self.project_out == False
        # heads가 1이 아니고 dim_head가 in_channel과 같지 않으면 self.project_out == True
        self.project_out  = not (heads == 1 and dim_head == in_channels)

        self.q_stride     = q_stride
        self.k_stride     = k_stride
        self.v_stride     = v_stride

        self.scale        = dim_head ** -0.5
        self.pad          = (kernel_size[0] - q_stride) // 2

        # 어텐션 모델을 위한 Q, K, V 정의
        self.Q = SeparableConv2D(
            in_channels = self.in_channels, 
            out_channels = self.out_channels, 
            kernel_size = self.kernel_size, 
            stride = self.q_stride,
            padding = self.pad)
        self.K = SeparableConv2D(
            in_channels = self.in_channels, 
            out_channels = self.out_channels, 
            kernel_size = self.kernel_size, 
            stride = self.k_stride,
            padding = self.pad)
        self.V = SeparableConv2D(
            in_channels = self.in_channels, 
            out_channels = self.out_channels, 
            kernel_size = self.kernel_size, 
            stride = self.v_stride,
            padding = self.pad)
        
        # self.project_out == True -> Linear / self.project_out == False -> Identity
        self.FFN = nn.Sequential(
            nn.Linear(self.out_channels, self.in_channels)) if self.project_out else nn.Identity()


    def forward(self, features1, features2):
        """
        표기법
        b : 배치 사이즈
        h : 헤드의 수
        d : 헤드의 디멘젼
        1. [B, HW, C]인 텐서를 받아서 다시 [B, C, H, W]으로 변환한다.
        2. Q, K, V에 해당하는 SepConv에서 다음의 텐서 형식으로 바뀐다.
           [B, C, H, W] -> DepthWise -> [B, C, H, W] -> PointWise -> [B, C', H, W] (C' = head * dim_head)
        3. [B, h * d, H, W]에서 [B, h, HW, d]으로 바꿈
        4. Q, K끼리 dot product하면 [B, h, HW, HW] 형태로 만듬. 다시 V와 dot procut하면 [B, h, HW, d]
        5. [B, head, HW, dim_head]를 [B, HW, head*dim_head]으로 바꿈
        6. 마지막 FFN [B, head*dim, C] weight matrix와 곱해서 [B, HW, C] 형태로 만듬

        Args:
            features1: [B, HW, C] ~ Q 
            features2: [B, HW, C] ~ K, V
        return:
            features: [B, HW, C]
        """
        features1 = rearrange(features1, 'b (l w) n -> b n l w', l = self.size[0], w = self.size[1])
        features2 = rearrange(features2, 'b (l w) n -> b n l w', l = self.size[0], w = self.size[1])

        query = self.Q(features1) 
        query = rearrange(query, 'b (h d) l w -> b h (l w) d', h = self.heads)
        key   = self.K(features2)
        key   = rearrange(key, 'b (h d) l w -> b h (l w) d', h = self.heads)
        value = self.V(features2)
        value = rearrange(value, 'b (h d) l w -> b h (l w) d', h = self.heads)

        # 채널 와이즈로 Q, K 곱해서 다시 [B head HW HW] 차원으로 만듬
        dot_product = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        attention   = dot_product.softmax(dim = -1)
        
        # 채널 와이즈로 Attention [B head HW HW] * Value [B head HW dim] = [B head HW dim_head]
        # [B head HW dim_head] -> [B HW head*dim_head]
        features = einsum('b h i j, b h j d -> b h i d', attention, value)
        features = rearrange(features, 'b h n d -> b n (h d)')
        features = self.FFN(features)
        return features