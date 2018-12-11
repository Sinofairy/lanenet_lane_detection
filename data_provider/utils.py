import os
from PIL import Image
import numpy as np


def enet_weighing(dataloader, num_classes, c=1.02):
    """计算每个类别的权重，是计算当前所有数据的权重
        w_class = 1 / (ln(c + p_class)),
    Args:
        dataloader ("data.Dataloader"): 数据加载器
        num_classes ("int"): 类别的数量
        c ("int", optional): 一个计算权重的超参数 Default: 1.02.
    """
    class_count = 0
    total = 0
    for _, label in dataloader:
        label = label.cpu().numpy()

        # 把tensor展平
        flat_label = label.flatten()

        # 计算每个label出现的次数
        class_count += np.bincount(flat_label, minlength=num_classes)
        # 计算当前batch中的所有像素点的数量
        total += flat_label.size

    # 计算权重score，然后计算每个类别的权重
    propensity_score = class_count / total
    class_weights = 1 / (np.log(c + propensity_score))

    return class_weights

"""
Computes class weights using median frequency balancing as described
in https://arxiv.org/abs/1411.4734:
"""
