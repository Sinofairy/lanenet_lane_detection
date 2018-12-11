from torch.autograd import Variable


class Test():
    """ 使用dataloader和loss criterion和特定的测试数据测试model

    Args:
        model ("nn.Module"): 实例网络
        data_loader ("Dataloader"): 迭代数据提供类
        criterion ("Optimizer"): loss criterion.
        metric ("Metric"): 指定要返回的instance metric
        use_cuda ("bool"): If "True", 是否使用GPU

    """
    def __init__(self, model, data_loader, criterion, metric, use_cuda):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.use_cuda = use_cuda

    def run_epoch(self, iteration_loss=True):
        """指行一个epoch的validation.

        Args:
            iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
            The epoch loss (float), and the values of the specified metrics

        """
        epoch_loss = 0.0
        self.metric.reset()
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs, labels = batch_data
            # labels = labels[:,0,:,:]

            # Wrap them in a Varaible
            inputs, labels = Variable(inputs), Variable(labels)
            if self.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                self.criterion = self.criterion.cuda()

            # Forward propagation
            outputs = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Keep track of loss for current epoch
            epoch_loss += loss.data[0]

            # Keep track of evaluation the metric
            self.metric.add(outputs.data, labels.data)

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.data[0]))

        return epoch_loss / len(self.data_loader), self.metric.value()
