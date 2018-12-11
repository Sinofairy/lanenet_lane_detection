import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

from models.deeplabv3 import DeepLabV3
import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data_provider.utils import enet_weighing
import utils
import numpy as np
# 获取参数
args = get_arguments()

# use_cuda = args.cuda and torch.cuda.is_available()

use_cuda = torch.cuda.is_available()

def load_dataset(dataset):
    print("\n加载数据...\n")

    print("选择的数据:", args.dataset)
    print("Dataset 目录:", args.dataset_dir)
    print("存储目录:", args.save_dir)

    # 数据转换和标准化
    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    # 转化:PILToLongTensor,因为是label，所以不能进行标准化
    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        ext_transforms.PILToLongTensor() #  (H x W x C) 转到 (C x H x W )
    ])

    # 获取选定的数据集
    # 加载数据集作为一个tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # 加载验证集作为一个tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=3,
        shuffle=True,
        num_workers=args.workers)

    # 加载测试集作为一个tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=3,
        shuffle=True,
        num_workers=args.workers)

    # 获取标签图像和RGB颜色中的像素值之间的编码
    class_encoding = train_set.color_encoding

    # 获取需要预测的类别的数量
    num_classes = len(class_encoding)

    # 打印调试的信息
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))

    # 展示一个batch的样本
    if args.mode.lower() == 'test':
        images, labels = iter(test_loader).next()
    else:
        images, labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    # 展示一个batch的samples和labels
    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # 获取类别的权重
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = 0
    if args.weighing.lower() == 'enet':
        # 传回的class_weights是一个list
        class_weights = np.array([1.44752114, 33.41317956, 43.89576605, 47.85765692, 48.3393951, 47.18958997, 40.2809274 , 46.61960781, 48.28854284])
        # class_weights = enet_weighing(train_loader, num_classes)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.Tensor(class_weights)
        # 把没有标记的类别设置为0
        # if args.ignore_unlabeled:
        #     ignore_index = list(class_encoding).index('unlabeled')
        #     class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # 初始化ENet
    model = DeepLabV3(num_classes)
    # model.load_state_dict(torch.load("save/model_13_2_2_2_epoch_580.pth"))
    # model.aspp.conv_1x1_4 = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    # 检查网络结构是否正确
    print(model)

    # 交叉熵的损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # 学习率衰减
    # lr_decay_epochs: 学习率衰减期。
    # lr_decay: 学习率衰减的乘积因子,默认值:-0.1
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # 评价指标
    # if not args.ignore_unlabeled:
    #     ignore_index = list(class_encoding).index('unlabeled')
    # else:
    #     ignore_index = None
    ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    if use_cuda:
        print("model使用GPU")
        model = model.cuda()
        criterion = criterion.cuda()

    # Optionally 从checkpoint恢复
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0

    # 开始 Training
    print()
    train = Train(model, train_loader, optimizer, criterion, metric, use_cuda)
    val = Test(model, val_loader, criterion, metric, use_cuda)
    for epoch in range(start_epoch, args.epochs):
        print(">>>> [Epoch: {0:d}] Training".format(epoch))

        lr_updater.step() # 修改学习率，开始训练
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)

        # 打印epoch，loss,mean iou
        print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch, epoch_loss, miou))

        # 如果当前的epochs结束，打印验证的进行一个验证
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))

            loss, (iou, miou) = val.run_epoch(args.print_step)

            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
                                      args)

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # 使用CrossEntropyLoss损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if use_cuda:
        criterion = criterion.cuda()

    # Evaluation metric
    # if not args.ignore_unlabeled:
    #     ignore_index = list(class_encoding).index('unlabeled')
    # else:
    #     ignore_index = None
    ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, use_cuda)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = Variable(images)
    if use_cuda:
        images = images.cuda()

    # Make predictions!
    predictions = model(images)

    #Predictions用"num_classes" channels进行one-hot encoded,使用maximum (1)的索引将其转换为单个int
    _, predictions = torch.max(predictions.data, 1) # max返回在通道维度的最大值的索引，也就是一维

    label_to_rgb = transforms.Compose([ # 把label转为RBG的图片，方便显示
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)



if __name__ == '__main__':

    # 检测数据集目录是否存在
    assert os.path.isdir(
        args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
            args.dataset_dir)

    # 存储的文件路径
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    # 导入需要的数据
    if args.dataset.lower() == 'lane':
        from data_provider.Lane_dataset import Lane_dataset as dataset

    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)
        if args.mode.lower() == 'full':
            test(model, test_loader, w_class, class_encoding)
    elif args.mode.lower() == 'test':
        # 初始化新的 ENet model
        num_classes = len(class_encoding)
        model = DeepLabV3(num_classes)
        # model.load_state_dict(torch.load("save/model_13_2_2_2_epoch_580.pth"))
        # model.aspp.conv_1x1_4 = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        if use_cuda:
            model = model.cuda()

        # 初始化优化器
        # checkpoint
        optimizer = optim.Adam(model.parameters())

        # 加载以前存储过的ENet模型
        model = utils.load_checkpoint(model, optimizer, args.save_dir,
                                      args.name)[0]
        print(model)
        test(model, test_loader, w_class, class_encoding)
    else:
        # Should never happen...but just in case it does
        raise RuntimeError(
            "\"{0}\" is not a valid choice for execution mode.".format(
                args.mode))
