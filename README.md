# CvT: Introducing Convolutions to Vision Transformers
Custom variation of CvT  
Existing CvT implementations have no consideration for Cross-Attention  
This repository shows CvT structure for self-attention and Cross-Attention.
In particular,  
Even if Cross-Attention maps of different sizes are injected, CA maps are interpolated to match the size of the SA feature maps.  
Paper URL  
1. [CvT: Introducing Convolutions to Vision Transforemrs](https://arxiv.org/abs/2103.15808)
# Usage
## Self-Attention of CvT
```
image   = torch.ones(size = [1, 8, 32, 80])
SelfCvT = SelfAttention_ConvolutionTransfomer(
                            self_feature_size = (32, 80), 
                            depth = 1, 
                            heads = 4, 
                            dim_head = 8, 
                            in_channels = 8, 
                            dim_mlp = 32) 
outputs = SelfCvT(image)

# Parameters
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         Rearrange-1              [-1, 2560, 8]               0
         LayerNorm-2              [-1, 2560, 8]              16
         LayerNorm-3              [-1, 2560, 8]              16
            Conv2d-4            [-1, 8, 32, 80]              80
       BatchNorm2d-5            [-1, 8, 32, 80]              16
            Conv2d-6           [-1, 32, 32, 80]             288
         SepConv2D-7           [-1, 32, 32, 80]               0
            Conv2d-8            [-1, 8, 32, 80]              80
       BatchNorm2d-9            [-1, 8, 32, 80]              16
           Conv2d-10           [-1, 32, 32, 80]             288
        SepConv2D-11           [-1, 32, 32, 80]               0
           Conv2d-12            [-1, 8, 32, 80]              80
      BatchNorm2d-13            [-1, 8, 32, 80]              16
           Conv2d-14           [-1, 32, 32, 80]             288
        SepConv2D-15           [-1, 32, 32, 80]               0
           Linear-16              [-1, 2560, 8]             264
ConvSelfAttention-17              [-1, 2560, 8]               0
      SelfPreNorm-18              [-1, 2560, 8]               0
        LayerNorm-19              [-1, 2560, 8]              16
           Linear-20             [-1, 2560, 32]             288
             GELU-21             [-1, 2560, 32]               0
           Linear-22              [-1, 2560, 8]             264
       FeedFoward-23              [-1, 2560, 8]               0
      SelfPreNorm-24              [-1, 2560, 8]               0
        Rearrange-25            [-1, 8, 32, 80]               0
================================================================
Total params: 2,016
Trainable params: 2,016
Non-trainable params: 0
----------------------------------------------------------------
```
## Cross-Attention of CvT
```
image1   = torch.ones(size = [1, 8, 32, 80])
image2   = torch.ones(size = [1, 8, 32, 80])
CrossCvT = CrossAttention_ConvolutionTransformer(
                            self_feature_size = (32, 80),
                            cross_faetures_size = (32, 80),
                            heads = 4,
                            dim_head = 8,
                            in_channels = 8,
                            dim_mlp = 32)
outputs  = CrossCvT(image1, image2)

# Parameters
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         Rearrange-1              [-1, 2560, 8]               0
         LayerNorm-2              [-1, 2560, 8]              16
         Rearrange-3               [-1, 640, 8]               0
         LayerNorm-4               [-1, 640, 8]              16
         LayerNorm-5              [-1, 2560, 8]              16
         LayerNorm-6               [-1, 640, 8]              16
            Conv2d-7            [-1, 8, 16, 40]              80
       BatchNorm2d-8            [-1, 8, 16, 40]              16
            Conv2d-9           [-1, 32, 16, 40]             288
        SepConv2D-10           [-1, 32, 16, 40]               0
           Conv2d-11            [-1, 8, 16, 40]              80
      BatchNorm2d-12            [-1, 8, 16, 40]              16
           Conv2d-13           [-1, 32, 16, 40]             288
        SepConv2D-14           [-1, 32, 16, 40]               0
           Conv2d-15            [-1, 8, 16, 40]              80
      BatchNorm2d-16            [-1, 8, 16, 40]              16
           Conv2d-17           [-1, 32, 16, 40]             288
        SepConv2D-18           [-1, 32, 16, 40]               0
           Linear-19               [-1, 640, 8]             264
ConvCrossAttention-20               [-1, 640, 8]               0
     CrossPreNorm-21               [-1, 640, 8]               0
        LayerNorm-22               [-1, 640, 8]              16
           Linear-23              [-1, 640, 32]             288
             GELU-24              [-1, 640, 32]               0
           Linear-25               [-1, 640, 8]             264
       FeedFoward-26               [-1, 640, 8]               0
      SelfPreNorm-27               [-1, 640, 8]               0
        Rearrange-28            [-1, 8, 16, 40]               0
================================================================
Total params: 2,048
Trainable params: 2,048
Non-trainable params: 0
----------------------------------------------------------------
```
## Example 1
```
Size of CA feature map != Size of SA feature map

image1   = torch.ones(size = [1, 8, 32, 80])
image2   = torch.ones(size = [1, 8, 16, 40])
CrossCvT = CrossAttention_ConvolutionTransformer(
                            self_feature_size = (32, 80),
                            cross_faetures_size = (16, 40),
                            heads = 4,
                            dim_head = 8,
                            in_channels = 8,
                            dim_mlp = 32)
outputs  = CrossCvT(image1, image2)
```
## Example 2
Heterogenous Network

image1  = torch.ones(size = [1, 8, 32, 32])
image2  = torch.ones(size = [1, 8, 32, 32])
SelfCvT = SelfAttention_ConvolutionTransfomer(
                            self_feature_size = (32, 32), 
                            depth = 1, 
                            heads = 4, 
                            dim_head = 8, 
                            in_channels = 8, 
                            dim_mlp = 64) 
CrossCvT = CrossAttention_ConvolutionTransformer(
                            self_feature_size = (32, 32),
                            cross_faetures_size = (32, 32),
                            heads = 4,
                            dim_head = 8,
                            in_channels = 8,
                            dim_mlp = 32)
hidden1 = SelfCvT(image1)
outputs = CrossCvT(hidden1, image2)
# Acknowledgement
Base CvT coe is borrowed from @rishikksh20 repo: https://github.com/rishikksh20/convolution-vision-transformers
