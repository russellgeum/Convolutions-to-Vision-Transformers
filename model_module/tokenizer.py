import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange



class TensorSplit(nn.Module):
    """
    [B C H W]의 피처 텐서나 이미지를 토큰화 해서 나누는 함수
    (가로부터 나누어서 나열)
    """
    def __init__(self, size, patch_size):
        super(TensorSplit, self).__init__()
        self.patch_height = patch_size[0]
        self.patch_width  = patch_size[1]
        self.height_unit  = int(size[0] / patch_size[0])
        self.width_unit   = int(size[1] / patch_size[1])

    def forward(self, images):
        feature_list = []
        for hi in range(self.height_unit):
            for wi in range(self.width_unit):
                patch = images[:, :, (hi)*self.patch_height : (hi+1)*self.patch_height, (wi)*self.patch_width : (wi+1)*self.patch_width]
                feature_list.append(patch)
        
        return feature_list


class TensorStack(nn.Module):
    """
    토큰화된 패치를 가로부터 다시 모아서 하나의 피처 텐서나 이미지로 복원하는 함수
    (가로부터 스택해서 복원)
    """
    def __init__(self, size, patch_size):
        super(TensorStack, self).__init__()
        self.patch_height = patch_size[0]
        self.patch_width  = patch_size[1]
        self.height_unit  = int(size[0] / patch_size[0])
        self.width_unit   = int(size[1] / patch_size[1])  

    def forward(self, features):
        B, C, _, _ = features[0].shape
        stacked_feature = torch.stack(features, dim = 0)
        stacked_feature = stacked_feature.reshape(self.height_unit, self.width_unit, B, C, self.patch_height, self.patch_width)

        stacked_feature = stacked_feature.permute(2, 3, 0, 4, 1, 5)
        stacked_feature = stacked_feature.reshape(B, C, self.height_unit * self.patch_height, self.width_unit * self.patch_width)
        return stacked_feature