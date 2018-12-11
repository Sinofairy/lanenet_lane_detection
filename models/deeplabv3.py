# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from .resnet import ResNet50_OS16
from .aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        self.resnet = ResNet50_OS16()

        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) 

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        # 此时的输出是原图的输出的缩小16倍
        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))
        # 双线性插值
        output = F.interpolate(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return output


if __name__ == "__main__":
    net = DeepLabV3(9)
    print(net)