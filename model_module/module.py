import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange

"""
사용 설명서
class SepConv2D: Q, K, V를 계산하는 클래스
class ConvSelfAttention: 이전 입력 모두를 Q, K, V에 사용하는 클래스
class ConvCrossAttention: 이전 입력을 Q, K에 사용하고 다른 피처를 V에 사용하는 클래스
class FeedForward: 어텐션의 출력을 FFN으로 transform하는 클래스
class SelfPreNorm
    LayerNorm을 SelfAttention하고 같이 사용
class CrossPreNOrm
    LayerNorm을 CrossAttention하고 같이 사용
"""

class SepConv2D(nn.Module):
    """
    Separatble Convoltuion
    """
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride = 1,
                padding = 0,
                dilation = 1):
        super(SepConv2D, self).__init__()
        """
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



class FeedFoward(nn.Module):
    """
    FeedForward Network
    """
    def __init__(self, in_channels ,dim_mlp):
        super(FeedFoward, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(in_channels, dim_mlp),
                        nn.GELU(),
                        nn.Linear(dim_mlp, in_channels))
    
    def forward(self, inputs):
        out = self.net(inputs)
        return out



class SelfPreNorm(nn.Module):
    """
    SelfPreNorm: 셀프 어텐션을 사용하는 노름 레이어
    PreNorm(dim, ConvSelfAttention(**args))
        self.norm     = nn.LayerNorm(dim)
        self.function = ConvSelfAttention(**args)
    """
    def __init__(self, in_channels, function):
        super(SelfPreNorm, self).__init__()
        self.norm     = nn.LayerNorm(in_channels)
        self.function = function

    def forward(self, x):
        out = self.function(self.norm(x))
        return out



class CrossPreNorm(nn.Module):
    """
    CrossPreNorm: 크로스 어텐션을 사용하는 노름 레이어
    PreNorm(dim, ConvCrossAttention(**args))
        self.norm     = nn.LayerNorm(dim)
        self.function = ConvCrossAttention(**args)
    """
    def __init__(self, in_channels, function):
        super(CrossPreNorm, self).__init__()
        self.norm     = nn.LayerNorm(in_channels)
        self.function = function

    def forward(self, x1, x2):
        out = self.function(self.norm(x1), self.norm(x2))
        return out



class SelfConvAttention(nn.Module):
    """
    Self Convolution Attention
    """
    def __init__(self, 
                 size,
                 in_channels, 
                 kernel_size = (3, 3), 
                 heads = 8, 
                 dim_head = 8,
                 q_stride = 1, 
                 k_stride = 1, 
                 v_stride = 1):
        super(SelfConvAttention, self).__init__()
        """
        Args
            size:  셀프 피처맵의 사이즈
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

        """
        어텐션 모델을 위한 Q, K, V 정의
        """
        self.Q = SepConv2D(
                    in_channels = self.in_channels, 
                    out_channels = self.out_channels, 
                    kernel_size = self.kernel_size, 
                    stride = self.q_stride,
                    padding = self.pad)
        self.K = SepConv2D(
                    in_channels = self.in_channels, 
                    out_channels = self.out_channels, 
                    kernel_size = self.kernel_size, 
                    stride = self.k_stride,
                    padding = self.pad)
        self.V = SepConv2D(
                    in_channels = self.in_channels, 
                    out_channels = self.out_channels, 
                    kernel_size = self.kernel_size, 
                    stride = self.v_stride,
                    padding = self.pad)
        
        # self.project_out == True -> Linear
        # self.project_out == False -> Identity
        self.FFN = nn.Sequential(
                    nn.Linear(self.out_channels, self.in_channels)) if self.project_out else nn.Identity()


    def forward(self, features):
        """
        Args:
            features: [B, HW, C]

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
        
        하씨 last_stage가 뭘 말하는거지? 그냥 쓰면 안되나?

        return:
            features: [B, HW, C]
        """
        batch, length, _, = *features.shape, 
        heads             = self.heads

        # last_stage가 True이면 이 조건문 실행 O, False이면 이 조건문 실행 X
        # print(features.shape)
        # if self.last_stage:
        #     cls_token = features[:, 0]
        #     print(cls_token.shape)
        #     features  = features[:, 1:]
        #     print(features.shape)
        #     print(cls_token.unsqueeze(1).shape)
        #     cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = heads)
        #     print(cls_token.shape)
        
        features = rearrange(features, 'b (l w) n -> b n l w', l = self.size[0], w = self.size[1])

        query    = self.Q(features) 
        query    = rearrange(query, 'b (h d) l w -> b h (l w) d', h = heads)
        key      = self.K(features)
        key      = rearrange(key, 'b (h d) l w -> b h (l w) d', h = heads)
        value    = self.V(features)
        value    = rearrange(value, 'b (h d) l w -> b h (l w) d', h = heads)

        # if self.last_stage:
        #     print(query.shape, key.shape, value.shape)
        #     query = torch.cat((cls_token, query), dim = 2)
        #     key   = torch.cat((cls_token, key), dim = 2)
        #     value = torch.cat((cls_token, value), dim = 2)

        # 채널 와이즈로 Q, K 곱해서 다시 [B head HW HW] 차원으로 만듬
        dot_product = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        attention   = dot_product.softmax(dim = -1)
        
        # 채널 와이즈로 Attention [B head HW HW] * Value [B head HW dim] = [B head HW dim_head]
        # [B head HW dim_head] -> [B HW head*dim_head]
        out = einsum('b h i j, b h j d -> b h i d', attention, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.FFN(out)
        return out


class CrossConvAttention(nn.Module):
    """
    CrossConvAttention
    """
    def __init__(self, 
                 size1, 
                 size2, 
                 in_channels,
                 kernel_size = (3, 3),
                 heads = 8, 
                 dim_head = 8, 
                 q_stride = 1,
                 k_stride = 1,
                 v_stride = 1):
        super(CrossConvAttention, self).__init__()
        """
        Args
            size1:  셀프 피처맵의 사이즈
            size2: 크로스 피처맵의 사이즈
            heads:              헤드의 갯수
            dim_head:           헤드 하나의 차원 수
            in_channels:        어텐션 모듈에 입력받을 피처맵의 채널 ([B, C, H, W]에서 C에 해당)
            kernel_size:        (kernel_height, kernel_width),
            q_stride, k_stride, v_stride: Q, K, V에서 작동할 컨볼루션의 스트라이드
        """
        self.size1 = size1
        self.size2 = size2
        self.in_channels  = in_channels
        self.kernel_size  = kernel_size

        self.heads        = heads
        self.dim_head     = dim_head
        self.out_channels = dim_head * heads
        
        # heads == 1이고 dim_head == in_channels이면 self.project_out == False
        self.project_out  = not (heads == 1 and dim_head == in_channels)

        self.q_stride     = q_stride
        self.k_stride     = k_stride
        self.v_stride     = v_stride

        self.scale        = dim_head ** -0.5
        self.pad          = (kernel_size[0] - q_stride) // 2

        """
        어텐션 모델 정의
        """
        self.Q = SepConv2D(
                    in_channels = self.in_channels, 
                    out_channels = self.out_channels, 
                    kernel_size = self.kernel_size, 
                    stride = self.q_stride,
                    padding = self.pad)

        self.K = SepConv2D(
                    in_channels = self.in_channels, 
                    out_channels = self.out_channels, 
                    kernel_size = self.kernel_size, 
                    stride = self.k_stride,
                    padding = self.pad)
        self.V = SepConv2D(
                    in_channels = self.in_channels, 
                    out_channels = self.out_channels, 
                    kernel_size = self.kernel_size, 
                    stride = self.v_stride,
                    padding = self.pad)
        
        # self.project_out == True -> Linear
        # self.project_out == False -> Identity
        self.FFN = nn.Sequential(
                    nn.Linear(self.out_channels, self.in_channels)) if self.project_out else nn.Identity()


    def forward(self, self_features, cross_features):
        """
        Args:
            self_features: [B, HW, C]
            cross_features: [B, HW, C]
        
        표기법
        b : 배치 사이즈
        h : 헤드의 수
        d : 헤드의 디멘젼
        작동 방식은 value가 cross_feature로 들어오는 것 제외하곤 SA와 동일

        return:
            features: [B, HW, C]
        """
        self_features  = rearrange(
                            self_features, 'b (l w) n -> b n l w', 
                            l = self.size1[0], 
                            w = self.size1[1])
        cross_features = rearrange(
                            cross_features, 'b (l w) n -> b n l w', 
                            l = self.size2[0], 
                            w = self.size2[1])

        
        # 향후 구현, 크로스 피처나 셀프 피처가 서로 크기 다른 경우 interpolate로 대응해야 함

        # cross_features = F.interpolate(input = cross_features,
        #                                 size = (self.size1[0], self.size1[1]), 
        #                                 mode = "bilinear",
        #                                 align_corners = True)
        # self_features  = F.interpolate(input = self_features,
        #                                 size = (self.size2[0], self.size2[1]), 
        #                                 mode = "bilinear",
        #                                 align_corners = True)


        query = self.Q(self_features) 
        query = rearrange(query, 'b (h d) l w -> b h (l w) d', h = self.heads)
        key   = self.K(cross_features)
        key   = rearrange(key, 'b (h d) l w -> b h (l w) d', h = self.heads)
        value = self.V(cross_features)
        value = rearrange(value, 'b (h d) l w -> b h (l w) d', h = self.heads)

        # 채널 와이즈로 Q, K 곱해서 다시 [B head HW HW] 차원으로 만듬
        dot_product = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        attention   = dot_product.softmax(dim = -1)
        
        # 채널 와이즈로 Attention [B head HW HW] * Value [B head HW dim_head] = [B head HW dim_head]
        # [B head HW dim_head] -> [B HW head*dim_head]
        out = einsum('b h i j, b h j d -> b h i d', attention, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.FFN(out)
        return out