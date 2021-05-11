import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from model_layer import *



class SelfAttention_ConvolutionTransfomer(nn.Module):
    def __init__(self, size, in_channels, depth = 1, heads = 8, dim_head = 8, dim_mlp = 64):
        super(SelfAttention_ConvolutionTransfomer, self).__init__()
        """
        Self-Attention을 사용하는 컨볼루션 트랜스포머
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
                            SelfConvAttention(
                                size = size, 
                                heads = heads, 
                                dim_head = dim_head, 
                                in_channels = in_channels)),
                    SelfPreNorm(in_channels, FeedFoward(in_channels, dim_mlp))]))

        self.inverse_shape = Rearrange('b (h w) c -> b c h w', h = size[0], w = size[1])

    def forward(self, features):
        """
        Args:
            features: [B C H W]

        returns:
            features: [B C H W]
        """
        features = self.forward_shape(features)
        features = self.LayerNorm(features)

        for attention, feedforward in self.module:
            features = attention(features) + features
            features = feedforward(features) + features

        features = self.inverse_shape(features)
        return features



class CrossAttention_ConvolutionTransformer(nn.Module):
    def __init__(self, size1, size2, in_channels, heads = 8, dim_head = 8, dim_mlp = 64):
        super(CrossAttention_ConvolutionTransformer, self).__init__()
        """
        Cross-Attention을 사용하는 컨볼루션 트랜스포머
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
                              CrossConvAttention(
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
            cross_features: [B, C, H, W]
        
        returns
            features: [B, C, H, W]
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



class LinearToken_ConvolutionTransformer(nn.Module):
    def __init__(self, size, patch_size, in_channels, embedding_dim, depth, heads = 8, dim_head = 8, dim_mlp = 64):
        super(LinearToken_ConvolutionTransformer, self).__init__()
        self.size          = size
        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.embedding_dim = embedding_dim
        self.depth         = depth
        self.heads         = heads
        self.dim_head      = dim_head
        self.dim_mlp       = dim_mlp

        self.patch_height = int(size[0]/patch_size[0])
        self.patch_width  = int(size[1]/patch_size[1])

        # [B C H W] -> [B (hw) embbeding_dim]
        self.embbedding = LinearPatchEmbbeding(
                                in_channels = self.in_channels, 
                                patch_size = self.patch_size, 
                                embbeding_dim = self.embedding_dim)
        self.layernorm  = nn.LayerNorm(self.embedding_dim)
        
        # [B (hw) embbeding_dim] -> [B embbeding_dim, h, w]
        self.rearrange  = Rearrange('b (h w) d -> b d h w', h = self.patch_height, w = self.patch_width)

        # 이 클래스에서는 self_attention의 in_channels가 embbeding_dim이고 size가 패치의 수가 됨 (자른 패치 사이즈에 대해서 임베딩을 이미 하였으므로)
        self.attention  = SelfAttention_ConvolutionTransfomer(
                                size = (self.patch_height, self.patch_width),
                                in_channels = self.embedding_dim,
                                depth = self.depth,
                                heads = self.heads,
                                dim_head = self.dim_head,
                                dim_mlp = self.dim_mlp)
    
    def forward(self, images):
        """
        Args:
            images: [B, C, H, W]

        1. Linear Embbeindg: [B, hw, emb_dim]
        2. rearrang:  [B, emb_dim, h, w]
        3. attention: [B, emb_dim, h, w]

        returns
            features: [B, emb_dim, h, w]
        """
        features = self.embbedding(images)
        features = self.layernorm(features)

        features = self.rearrange(features)
        features = self.attention(features)
        return features



class ConvToken_ConvolutionTransformer(nn.Module):
    def __init__(self, size, patch_size, in_channels, embedding_dim, depth = 1, heads = 8, dim_head = 8, dim_mlp = 64):
        super(ConvToken_ConvolutionTransformer, self).__init__()
        self.size          = size
        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.embedding_dim = embedding_dim
        self.depth         = depth
        self.heads         = heads
        self.dim_head      = dim_head
        self.dim_mlp       = dim_mlp

        self.patch_height = int(size[0]/patch_size[0])
        self.patch_width  = int(size[1]/patch_size[1])

        self.embedding = ConvPatchEmbbeding(
                            in_channels = self.in_channels,
                            patch_size = self.patch_size,
                            embbeding_dim = self.embedding_dim)
        self.layernorm = nn.LayerNorm(self.embedding_dim)

        self.rearrange = Rearrange('b (h w) d -> b d h w', h = self.patch_height, w = self.patch_width)
        self.attention = SelfAttention_ConvolutionTransfomer(
                            size = (self.patch_height, self.patch_width),
                            in_channels = self.embedding_dim,
                            depth = self.depth,
                            heads = self.heads,
                            dim_head = self.dim_head,
                            dim_mlp  = self.dim_mlp)
    
    def forward(self, images):
        """
        Args:
            images: [B, C, H, W]

        returns
            features: [B, emb_dim, h, w]
        """
        features = self.embedding(images)
        features = self.layernorm(features)

        features = self.rearrange(features)
        features = self.attention(features)
        return features

if __name__ == "__main__":
    images = torch.ones([4, 3, 64, 64])
    layer  = ConvToken_ConvolutionTransformer((64, 64), (8, 4), 3, 64, 1, 8, 8, 64)
    output = layer(images)
    print(output.shape)