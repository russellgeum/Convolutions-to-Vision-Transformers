# CvT: Introducing Convolutions to Vision Transformers
Custom variation of Convolution to Vision Transformers  
Existing CvT implementations have no consideration for Cross-Attention  
This repository shows CvT structure for self-attention and Cross-Attention.
In particular,  
Even if Cross-Attention maps of different sizes are injected,  
CA maps are interpolated to match the size of the SA feature maps.  
Paper URL  
- [CvT: Introducing Convolutions to Vision Transforemrs](https://arxiv.org/abs/2103.15808)  
  
# Hierarchy 
```
ㄴmodel_module
    embedding.py
        class LinearPatchEmbbeding
        class ConvPatchEmbbeding
    tokenizer.py
        class TensorSplit
        class TensorStack
    module.py
        class ConvSelfAttention
        class ConvCrossAttention
ㄴmodel_layer.py
        class SelfAttention_ConvolutionTransfomer
        class CrossAttention_ConvolutionTransformer
        class TokenSelfAttention_ConvolutionTransformer
        class TokenCrossAttention_ConvolutionTransformer
``` 
# Usage
## Self-Attention of CvT
```
image1  = torch.ones([8, 3, 64, 128])
layer1  = SelfAttention_ConvolutionTransfomer(
                           size = (64, 128),
                           in_channels = 3,  
                           depth = 1, 
                           heads = 8, 
                           dim_head = 8, 
                           dim_mlp = 64) 
result  = layer1(image1)

# image1 shape
torch.Size([8, 3, 64, 128])

# result shape
torch.Size([8, 3, 64, 128])

# Parameters
1540
```
  
## Cross-Attention of CvT
```
image1  = torch.ones([8, 3, 64, 128])
image2  = torch.ones([8, 3, 64, 128])
layer2  = CrossAttention_ConvolutionTransformer(
                           size1 = (64, 128),
                           size2 = (64, 128), 
                           in_channels = 3,
                           heads = 8,
                           dim_head = 8, 
                           dim_mlp = 64)
result  = layer2(image1, image2)

# image1, image2 shape
torch.Size([8, 3, 64, 128])
torch.Size([8, 3, 64, 128])

# result shape
torch.Size([8, 3, 64, 128])

# Parameters
1540
```
  
## Tokenizer Self-Attention of CvT
```
image1  = torch.ones([8, 3, 64, 128])
layer3  = TokenSelfAttention_ConvTransformer(
                           size = (64, 128),
                           patch_size = (16, 32),
                           in_channels = 3,  
                           depth = 1, 
                           heads = 8, 
                           dim_head = 8, 
                           dim_mlp = 64) 
result  = layer3(image1)

# image1 shape
torch.Size([8, 3, 64, 128])

# result shape
torch.Size([8, 3, 64, 128])

# Parameters
24640
```
  
## Tokenizer Cross-Attention of CvT
```
image1  = torch.ones([8, 3, 64, 128])
image2  = torch.ones([8, 3, 64, 128])
layer4  = TokenCrossAttention_ConvTransformer(
                           size1 = (64, 128),
                           size2 = (64, 128), 
                           patch_size = (16, 32),
                           in_channels = 3,
                           heads = 8,
                           dim_head = 8, 
                           dim_mlp = 64)
result  = layer4(image1, image2)

# image1, image2 shape
torch.Size([8, 3, 64, 128])
torch.Size([8, 3, 64, 128])

# result shape
torch.Size([8, 3, 64, 128])

# Parameters
24640
```
  
# Acknowledgement  
Base CvT code is borrowed from @rishikksh20 repo: https://github.com/rishikksh20/convolution-vision-transformers  
  
# 향후 대응할 목록  
1. 크로스 어텐션에서 서로 다른 사이즈의 텐서가 들어왔을 경우  
    A1) 셀프 텐서에 크로스 텐서 사이즈를 맞추기  
    A2) 크로스 텐서에 셀프 텐서 사이즈를 맞추기  
    A3) 둘 다 대응할 수 있고, F.interpolate로 대응  
2. 네이밍 규칙 정리하기  
가장 작은 단위: Self_ConvAttention, Cross_ConvAttention  
SCA로만 이루어진 트랜스포머?  -> SelfAttention-ConvolutionTransformer  
CCA - SCA로 이루어진 트랜스포머?  -> CrossAttention-ConvolutionTransformer  