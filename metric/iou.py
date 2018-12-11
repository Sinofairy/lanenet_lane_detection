import torch
import numpy as np
from metric import metric
from metric.confusionmatrix import ConfusionMatrix


class IoU(metric.Metric):
    """计算每个类的并集(IoU)和相应的均值(mIoU)
    联合交叉(IoU)是语义的通用评估度量分割,预测首先在混淆矩阵中累积
    IoU的计算方法如下:

    IoU = true_positive /(true_positive + false_positive + false_negative)

    Args:
        num_classes(int): 分类问题中的类数
        normalized(boolean,optional): 确定是否混淆,矩阵是否归一化, 默认值: False
        ignore_index(int或iterable，optional): 要忽略的类的索引,在计算IoU时, 可以是int,也可以是任何可迭代的int。
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """将predicted和target添加到IoU metric.混淆矩阵

        Args:
            predicted (Tensor): 可以是(N, K, H, W) tensor，从N个样本的K类的类别得分
                                或者是(N, H, W) tensor，值在0到K-1
            target (Tensor): 可以是N个样本和K类的目标分数的(N,K,H,W)张量,或者(N,H,W) tensor 值在0到K-1
        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """计算 IoU 和 mean IoU.

        平均计算忽略IoU阵列的NaN元素。

        Returns:
            Tuple: (IoU, mIoU). 他的第一个输出是每个类IoU，对于K类，它是带有K个元素的numpy.ndarray 第二个输出是平均IoU。
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)
