# CvT: Introducing Convolutions to Vision Transformers
Custom variation of transformer layer based on Convolution to Vision Transformers.  
This repository shows CvT structure for self-attention and Cross-Attention or Embedding Layer.  
In future, This repository will support SwinTransformer structure.  
Reference Paper URL  
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
- [CvT: Introducing Convolutions to Vision Transforemrs](https://arxiv.org/abs/2103.15808)  
# Folder 
```
ㄴmodel_layer
    ㄴtokenizer.py
        class ImageTokenizer
        class ImageStacker
    ㄴembedding.py
        class LinePatchEmbbeding
        class ConvPatchEmbbeding
    ㄴattention.py
        class SelfConvAttention
        class CrossConvAttention
    ㄴtransforemr.py
        class LineTokenConvTransformer  
        class ConvTokenConvTransformer  
        class SelfConvTransfomer  
        class CrossConvTransformer  
``` 
# Usage
## CvT with Linear Tokenizer  
```
tensor = torch.ones([8, 3, 16, 16])
print(tensor.shape) # torch.Size([8, 3, 16, 16])

layer   = LineTokenConvTransformer((16, 16), (4, 4), 3, 3)
outputs = layer(tensor)

print(outputs.shape) # torch.Size([8, 3, 4, 4])
```
## CvT with Conv Tokenizer  
```
tensor = torch.ones([8, 3, 16, 16])
print(tensor.shape) # torch.Size([8, 3, 16, 16])

layer   = ConvTokenConvTransformer((16, 16), (4, 4), 3, 3)
outputs = layer(tensor)

print(outputs.shape) # torch.Size([8, 3, 4, 4])
```
## Self-Attention of CvT (SA-CvT)  
```
tensor = torch.ones([8, 3, 16, 16])
print(tensor.shape) # torch.Size([8, 3, 16, 16])

layer   = SelfConvTransfomer((16, 16), 3, 2)
outputs = layer(tensor) # torch.Size([8, 3, 16, 16])

print(outputs.shape)
```
## Cross-Attention of CvT (CA-CvT)  
```
tensor1 = torch.ones([8, 3, 16, 16])
tensor2 = torch.ones([8, 3, 16, 16])
print(tensor1.shape, tensor2.shape) # torch.Size([8, 3, 16, 16])

layer   = CrossConvTransformer((16, 16), 3, 1)
outputs = layer(tensor1, tensor2)
print(outputs.shape) # torch.Size([8, 3, 16, 16])
```
# Acknowledgement  
Base CvT code is borrowed from @rishikksh20  
repo: https://github.com/rishikksh20/convolution-vision-transformers  
Base Embedding code is borrowed from @FrancescoSaverioZuppichini  
repo: https://github.com/FrancescoSaverioZuppichini/ViT  
# 향후 대응할 목록  
2021-06-17 업데이트  
1. SwinTransformer와 통합  
2. 크로스 어텐션에서 서로 다른 사이즈의 텐서가 들어왔을 경우  
    A1) 셀프 텐서에 크로스 텐서 사이즈를 맞추기  
    A2) 크로스 텐서에 셀프 텐서 사이즈를 맞추기  
    A3) 둘 다 대응할 수 있고, F.interpolate로 대응 또는 할 필요가 있을까?  
        아직은 하지 않는 것이 무게를 두지만, 향후 NAS-aware한 연산에서는 필요할 수도 있을듯...  