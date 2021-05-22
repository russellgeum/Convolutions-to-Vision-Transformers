# CvT: Introducing Convolutions to Vision Transformers
Custom variation of transformer layer based on Convolution to Vision Transformers.  
Existing CvT implementations have no consideration for Cross-Attention.  
This repository shows CvT structure for self-attention and Cross-Attention.
In particular, (X, Not yet...)  
Even if Cross-Attention maps of different sizes are injected,  
CA maps are interpolated to match the size of the SA feature maps.  
Reference Paper URL  
- [CvT: Introducing Convolutions to Vision Transforemrs](https://arxiv.org/abs/2103.15808)  
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
  
# Folder 
```
ㄴmodel_layer/
    embedding.py
        class LinearPatchEmbbeding
        class ConvPatchEmbbeding
    tokenizer.py
        class ImageTokenizer
        class ImageStacker
    module.py
        class SelfConvAttention
        class CrossConvAttention
ㄴmodel_module.py  
        class SelfAttention_ConvolutionTransformer  
        class CrossAttention_ConvolutionTransformer  
        class LinearToken_ConvolutionTransformer  
        class ConvToken_ConvolutionTransformer  
``` 
# Usage
## Self-Attention of CvT (SA-CvT)  
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
  
## Cross-Attention of CvT (CA-CvT)  
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
  
## CvT with Linear Tokenizer  
```
images = torch.ones([4, 3, 64, 64])
layer  = LinearToken_ConvolutionTransformer((64, 64), (8, 4), 3, 64, 1, 8, 8, 64)
output = layer(images)

# images shape
torch.Size([4, 3, 64, 64])

# output shape
torch.Size([4, 64, 8, 16])
```
  
## CvT with Conv Tokenizer  
```
images = torch.ones([4, 3, 64, 64])
layer  = ConvToken_ConvolutionTransformer((64, 64), (8, 4), 3, 64, 1, 8, 8, 64)
output = layer(images)

# images shape
torch.Size([4, 3, 64, 64])

# output shape
torch.Size([4, 64, 8, 16])
```
  
# Acknowledgement  
Base CvT code is borrowed from @rishikksh20  
repo: https://github.com/rishikksh20/convolution-vision-transformers  
Base Embedding code is borrowed from @FrancescoSaverioZuppichini  
repo: https://github.com/FrancescoSaverioZuppichini/ViT  
  
# 향후 대응할 목록  
2021-05-11  
토크나이저를 리니어 매핑과 컨보 매핑 레이어로 임베딩하는 형태로 교체 (이것이 올바름)  
1. 크로스 어텐션에서 서로 다른 사이즈의 텐서가 들어왔을 경우  
    A1) 셀프 텐서에 크로스 텐서 사이즈를 맞추기  
    A2) 크로스 텐서에 셀프 텐서 사이즈를 맞추기  
    A3) 둘 다 대응할 수 있고, F.interpolate로 대응  
    할 필요가 있을까?  
2. 네이밍 규칙 정리하기  
3. SwinTransformer와 통합  
