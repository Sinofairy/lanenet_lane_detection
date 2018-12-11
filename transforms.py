import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision.transforms import ToPILImage


class PILToLongTensor(object):
    """Converts a "PIL Image" to a "torch.LongTensor".

    (H x W x C) 转到 (C x H x W )
    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor

    """

    def __call__(self, pic):
        """Performs the conversion from a "PIL Image" to a "torch.LongTensor".

        Keyword arguments:
        - pic ("PIL.Image"): the image to convert to "torch.LongTensor"

        Returns:
        A "torch.LongTensor".
        """
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            # (C x H x W )转到(H x W x C)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel) # 改变形状以后返回

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0,2).contiguous().long().squeeze_()


class LongTensorToRGBPIL(object):
    """用于在预测的时候把对应类别的输出加上RGB的颜色，方便显示
     转换 "torch.LongTensor" 到"PIL image".

    输入是一个"torch.LongTensor"，其中每个像素的值标识该类.

    Args:
        rgb_encoding ("OrderedDict"): 一个"OrderedDict"包含了 pixel values, class names, and class colors.

    """
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """把"torch.LongTensor" 转换为 "PIL image"
        Args:
            tensor ("torch.LongTensor"): 要转换的tensor
        Returns:
            A "PIL.Image".

        """
        # 检查 label_tensor 是否为LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}"
                            .format(type(tensor)))
        # 检查 encoding 是否为 ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(
                type(self.rgb_encoding)))

        #label_tensor可能是没有通道尺寸的图像，在这种情况下会解压缩它,增加一个channel
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)

        # 创建一个3channel的color map
        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # 获取等于index的元素mask，维度和tensor一样，也就是等于的话值为1，不是就为0，如果维度有1的，那就压缩
            mask = torch.eq(tensor, index).squeeze_()
            # 填充color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value) # 根据维度补值

        return ToPILImage()(color_tensor)
