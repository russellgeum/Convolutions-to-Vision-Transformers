import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange



# 피처 텐서의 토큰을 모양을 유지하지 않고, 펼쳐서 2D matrix 형태로 만들거나
# 그 2D matrix를 다시 모아, 피처 텐서로 만드는 함수
class ImageTokenizer(nn.Module):
    """
    [B C H W]의 피처 텐서나 이미지를 토큰화해서
    세로는 토큰의 수, 가로로는 패치 높이 * 패치 사이즈로 변환한 매트릭스로 보내는 레이어
    """
    def __init__(self, size, patch_size):
        super(ImageTokenizer, self).__init__()
        self.patch_height = patch_size[0]
        self.patch_width  = patch_size[1]
        self.height_unit  = int(size[0] / patch_size[0])
        self.width_unit   = int(size[1] / patch_size[1])
        self.rearrange    = Rearrange('b c (h p1) (w p2) -> b c (h w) (p1 p2)', 
                                        p1 = self.patch_height, p2 = self.patch_width)

    def forward(self, inputs):
        """
        Args:
            inputs: [B, C, H, W]
        returns
            outputs: [B, C, num_of_patches, patch_height * patch_width]
        """
        outputs = self.rearrange(inputs)
        return outputs



class ImageStacker(nn.Module):
    def __init__(self, size, patch_size):
        super(ImageStacker, self).__init__()
        self.patch_height = patch_size[0]
        self.patch_width  = patch_size[1]
        self.height_unit  = int(size[0] / patch_size[0])
        self.width_unit   = int(size[1] / patch_size[1])
        self.rearrange    = Rearrange('b c (h w) (p1 p2) ->b c (h p1) (w p2)', 
                                        h = self.height_unit, w = self.width_unit,
                                        p1 = self.patch_height, p2 = self.patch_width)
        
    def forward(self, inputs):
        """
        Args:
            inputs: [B, C, num_of_patches, patch_height * patch_width]
        returns
            outputs: [B, C, H, W]
        """
        outputs = self.rearrange(inputs)
        return outputs