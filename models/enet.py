import torch.nn as nn
import torch
from torch.autograd import Variable


class InitialBlock(nn.Module):
    """初始块由两个分支组成: 
    1.主要分支，用步幅2进行定期卷积;
    2.执行最大池化的扩展分支

    并行执行两个操作并连接结果
    允许有效的下采样和扩展 
    主要分支输出13个feature maps，而扩展分支输出3，
    连接后总共16个feature maps

    Args:
        in_channels (int): 输入通道
        out_channels (int): 输出通道
        kernel_size (int, optional): 卷积核的大小 Default: 3.
        padding (int, optional): padding的大小. Default: 0.
        bias (bool, optional): 是否使用bias. Default: False.
        relu (bool, optional): 当 "True"的时候，使用relu作为激活函数，否则就使用prelu，Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=0,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 主要的分支
        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias)

        # 边缘分支
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)

        # 归一化
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU layer
        self.out_prelu = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        # 把上面的结果连接起来
        out = torch.cat((main, ext), 1) # 在列上进行连接
        out = self.batch_norm(out)
        return self.out_prelu(out)


class RegularBottleneck(nn.Module):
    """Regular bottleneck是ENet的主要组成部分
    Main branch:
    1. Shortcut connection.

    Extension branch:
    1. 1x1卷积，通过"internal_ratio"减少通道数，也称为投影;
    2.规则的，扩张的或不对称的卷积;
    3. 1x1卷积，将通道数量增加回"通道"，也称为扩展;
    4. dropout作为regular

    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        """
        Args:
            channels: 输入的深度
            internal_ratio: 深度除以投影比率
        """

        # 
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))

        # 初始1x1投影的深度减少量 深度除以投影比率 默认值为4
        internal_channels = channels // internal_ratio # 相除向下取整，得到投影后的通道数量

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - shortcut connection

        # 边缘分支，1x1卷积投影，然后一个regular，dilated或者非对称卷积，
        # 然后又是一个1x1的卷积，最后一个regularizer(spatial dropout)
        # channels的数量是固定的

        # 1x1 卷积投影
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels, # internal_channels是投影后的通道数量
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 如果卷积是asymmetric非对称卷积，要把卷积拆分为两次卷积
        # 比如: 5x5拆分为5x1,然后是1x5的卷积
        if asymmetric: # 两次asymmetric卷积之后的通道数量的一样
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation,
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)
        else: # 否则就是正常的卷积
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 升维
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels, # 输出的通道数量和当前bottleneck的输入通道数量一样
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob) # 随机丢弃

        # PReLU layer 两个分支相加以后进行一个激活
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # main和extension branches相加
        out = main + ext

        return self.out_prelu(out)


class DownsamplingBottleneck(nn.Module):
    """下采样瓶颈进一步下采样特征图大小

    Main branch:
    1.max pooling(2x2);保存索引以用于以后的上采样

    Extension branch:
    1. 2x2卷积与步幅2，通过"internal_ratio"减少通道数，也称为投影;
    2. 常规卷积(默认为3x3);
    3. 1x1卷积，增加了"out_channels"的通道数，也称为扩展;
    4. dropout 是 regularizer.

    Args:   
        in_channels(int): 输入通道的数量
        out_channels(int): 输出通道的数量
        internal_ratio(int，optional): 应用于"channels"的比例因子，用于计算投影后的通道数例如特定"channels"等于128，internal_ratio等于2投影后的通道数为64.默认值: 4
        kernel_size(int，optional): 扩展分支第2项中描述的卷积层中使用的过滤器的内核大小默认值: 3
        padding(int，optional): 将零填充添加到输入的两侧默认值: 0
        dilation(int,optional): 扩展分支第2项中描述的卷积的内核元素之间的间距默认值: 1
        asymmetric(bool，optional): 如果扩展分支的第2项中描述的卷积是不对称的，则标记默认值: False
        return_indices(bool,optional): 如果"True"，将返回最大索引和输出以后解放时很有用
        dropout_prob(float,optional): 元素归零的概率默认值: 0(无丢失)
        bias(bool，optional): 如果为"True"，则为输出添加可学习的偏差默认值: False
        relu(bool，optional): 当"True"ReLU用作激活函数时;否则，使用PReLU默认值: True

    """

    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3,
                 padding=0, return_indices=False, dropout_prob=0, bias=False, relu=True):
        super().__init__()

        # 存储以后需要的参数,max pooling的索引
        self.return_indices = return_indices

        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(kernel_size, stride=2,
                                      padding=padding, return_indices=return_indices)

        # Extension branch - 2x2 卷积,然后一个regular,dilated或者asymmetric卷积, 
        # 然后是另外一个1x1卷积. channels的数量翻倍

        # 2x2 stride=2，投影
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=2,
                      stride=2, bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # Convolution普通的卷积
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size,
                      stride=1, padding=padding,
                      bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 扩展卷积
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1,
                      stride=1, bias=bias), nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = Variable(torch.zeros(n, ch_ext - ch_main, h, w))

        #在连接之前检查，检查main数据是否为GPU的数据还是CPU数据，然后直接padding
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate 把main的维度padding以后和ext一样，好进行加和，对应的数据相加
        main = torch.cat((main, padding), 1) # 在通道的维度相连

        # Add
        out = main + ext

        return self.out_prelu(out), max_indices


class UpsamplingBottleneck(nn.Module):
    """upsampling bottlenecks使用从相应的downsampling bottleneck存储的max pooling索引对上映特征映射分辨率进行上采样

    Main branch:
    1. 1x1卷积，步幅1通过"internal_ratio"减少通道数，也称为投影;
    2.使用来自相应下采样最大池层的最大池索引的最大unpool层

    扩展分支: 
    1. 1x1卷积，步幅1，通过"internal_ratio"减少通道数，也称为投影;
    2. 转置卷积(默认为3x3);
    3. 1x1卷积，将通道数增加到"out_channels"，也称为扩展;
    4. dropout是regular

    Args:  
        in_channels(int): 输入通道的数量
        out_channels(int): 输出通道的数量
        internal_ratio(int，optional): 应用于"in_channels"的比例因子，用于计算投影后的通道数例如特定
                                       "in_channels"等于128，"internal_ratio"等于2投影后的通道数为64.默认值: 4
        kernel_size(int，optional): 扩展分支第2项中描述的卷积层中使用的过滤器的内核大小默认值: 3
        padding(int，optional): 将零填充添加到输入的两侧默认值: 0
        dropout_prob(float，optional): 元素归零的概率默认值: 0(无丢失)
        bias(bool，optional): 如果为"True"，则向输出添加可学习的偏差.Default: False
        relu(bool，可选): 当"True"ReLU用作激活功能时;否则，使用PReLU默认值: True

    """

    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3,
                 padding=0, dropout_prob=0, bias=False, relu=True):
        super().__init__()

        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling后面跟feature map(channels)的padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # the max pooling layers
        # max pooling的窗口移动的步长的默认值是kernel_size
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch 1x1的卷积，然后是普通，dilated或asymmetric卷积，然后是另一个1x1卷积
        # channels翻倍

        # 1x1投影 stride = 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels,
                      kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation)

        # Transposed卷积
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1, # 此处要对输出的每条边补充0的层数
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion卷积
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU
        self.out_prelu = activation

    def forward(self, x, max_indices):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add
        out = main + ext

        return self.out_prelu(out)


class ENet(nn.Module):
    """生成ENet模型。

     Args: 
        num_classes(int): 类数
        encoder_relu(bool，可选): 当"True"时，ReLU用作编码器块/层中的激活功能; 否则，使用PReLU。 默认值：False
        decoder_relu(bool，optional): 当"True"时，ReLU用作解码器块/层中的激活函数; 否则，使用PReLU。 默认值：True

    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(3, 16, padding=1, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            padding=1,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64, # channel增加了，但是kennels减小了
            128,
            padding=1,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)

        return x
