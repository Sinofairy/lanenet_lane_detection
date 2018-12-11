import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

"""
output stride
输出步幅，输入图像分辨率与最终输出分辨率之比（OS16：输出大小为（img_h / 16，img_w / 16））
（OS8：输出大小为（img_h / 8，img_w / 8））
"""
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) 
    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels
    layer = nn.Sequential(*blocks) 
    return layer


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        

        out = F.relu(self.bn1(self.conv1(x))) 
        out = F.relu(self.bn2(self.conv2(out))) 
        out = self.bn3(self.conv3(out)) 
        out = out + self.downsample(x) 
        out = F.relu(out) 
        return out

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_Bottleneck_OS16, self).__init__()

        if num_layers == 50:
            resnet = models.resnet50()  
            resnet.load_state_dict(torch.load("save/resnet/resnet50-19c8e357.pth"))   
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
            print ("pretrained resnet, 50")
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)
        output = self.layer5(c4)
        return output

def ResNet50_OS16():
    return ResNet_Bottleneck_OS16(num_layers=50)
