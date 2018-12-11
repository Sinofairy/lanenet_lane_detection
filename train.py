from torch.autograd import Variable


class Train():
    """根据提供的training dataset dataloader,optimizer, 和loss criterion.进行训练

    Args:
        model ("nn.Module"): 实例网络
        data_loader ("Dataloader"): 迭代数据提供类
        optim ("Optimizer"): optimization algorithm.
        criterion ("Optimizer"): loss criterion.
        metric ("`Metric"): 指定要返回的instance metric
        use_cuda ("bool"): If "True", 是否使用GPU
    """

    def __init__(self, model, data_loader, optim, criterion, metric, use_cuda):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.use_cuda = use_cuda

    def run_epoch(self, iteration_loss=True):
        """一个epoch的训练

        Args:
            iteration_loss ("bool", optional): 每步打印损失函数
        Returns:
            epoch loss

        """
        epoch_loss = 0.0
        self.metric.reset() # 混淆矩阵全部置0
        if self.use_cuda:
            print("当前正在使用GPU")
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs, labels = batch_data
            # labels = labels[:,0,:,:]
            # Wrap them in a Varaible
            inputs, labels = Variable(inputs), Variable(labels)
            if self.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                # self.criterion = self.criterion.cuda()

            # Forward propagation
            outputs = self.model(inputs)

            # Loss computation
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # 跟踪当前epoch的损失
            epoch_loss += loss.data[0]

            # 跟踪当前的metric
            self.metric.add(outputs.data, labels.data)

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.data[0]))

        return epoch_loss / len(self.data_loader), self.metric.value()
