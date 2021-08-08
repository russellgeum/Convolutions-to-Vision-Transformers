# Introducing Vision Transformers
Implementation vision transformer module (specially, attention layer)  
This repository gives vision attention or embedding Layer.  
Reference Paper  
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)  
- [CvT: Introducing Convolutions to Vision Transforemrs](https://arxiv.org/abs/2103.15808)  
- [Contextual Transformer Networks for Visual Recognition](https://arxiv.org/abs/2107.12292)  
# Folder 
```
ㄴmodel_layer
    ㄴtokenizer.py
        class ImageTokenizer
        class ImageStacker
    ... ...
    ㄴtransforemr.py
        class LineTokenConvTransformer  
        class ConvTokenConvTransformer  
        class SelfConvTransfomer  
        class CrossConvTransformer
``` 
# Usage
## CvT with Linear Tokenizer  
```
tensor = torch.ones([8, 3, 16, 16]) # torch.Size([8, 3, 16, 16])
layer   = LineTokenConvTransformer((16, 16), (4, 4), 3, 3)
outputs = layer(tensor)             # torch.Size([8, 3, 4, 4])
```
## CvT with Conv Tokenizer  
```
tensor = torch.ones([8, 3, 16, 16]) # torch.Size([8, 3, 16, 16])
layer   = ConvTokenConvTransformer((16, 16), (4, 4), 3, 3)
outputs = layer(tensor)             # torch.Size([8, 3, 4, 4])
```
## Self-Attention of CvT (SA-CvT)  
```
tensor = torch.ones([8, 3, 16, 16]) # torch.Size([8, 3, 16, 16])
layer   = SelfConvTransfomer((16, 16), 3, 2)
outputs = layer(tensor)             # torch.Size([8, 3, 16, 16])

print(outputs.shape)
```
## Cross-Attention of CvT (CA-CvT)  
```
tensor1 = torch.ones([8, 3, 16, 16]) # torch.Size([8, 3, 16, 16])
tensor2 = torch.ones([8, 3, 16, 16]) # torch.Size([8, 3, 16, 16])

layer   = CrossConvTransformer((16, 16), 3, 1)
outputs = layer(tensor1, tensor2)    # torch.Size([8, 3, 16, 16])
```
# Acknowledgement  
Base CvT code is borrowed from @rishikksh20  
repo: https://github.com/rishikksh20/convolution-vision-transformers  
Base Embedding code is borrowed from @FrancescoSaverioZuppichini  
repo: https://github.com/FrancescoSaverioZuppichini/ViT  
# 향후 대응할 목록  
1. SwinTransformer 구현  
2. 몇 가지 가능한 Convolutional Attention 구현  
