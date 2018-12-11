import numpy as np
import torch
from metric import metric


class ConfusionMatrix(metric.Metric):
    """为多类分类问题构造一个混淆矩阵，便于计算iou
    Args:
        num_classes(int): 分类问题中的类数。
        normalized(boolean,optional): 确定是否混淆，矩阵是否归一化,默认值: False。

    参考: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """
    def __init__(self, num_classes, normalized=False):
        super().__init__()

        # 生成 num_classes*num_classes的矩阵
        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """
        混淆矩阵全部置0
        """
        self.conf.fill(0)

    def add(self, predicted, target):
        """计算混淆矩阵
        混淆矩阵的大小为K*K,K是类别数量

        Args:
            predicted (Tensor or numpy.ndarray): 可以是从N个样本和K类的模型获得的N×K张量的预测分数，或者是0和K-1之间的整数值的N-tensor。
            target (Tensor or numpy.ndarray): 对于N个样本和K类，可以是N×K的tensor的ground-truth，或者是0和K-1之间的N-tensor的整数值。

        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted): # 转化为ndarray的数据
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1: # 判断predicted的维数
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1) # 返回维度1的最大值的索引值
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together 
        # 此时的predicted的维数为1的array
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            K行K列的混淆矩阵, 行对应的是ground-truth targets, 列对应的是相应的预测
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf
