import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os


def batch_transform(batch, transform):
    """把一个batch的样本应用于transform
    Args:
        batch (): a batch的样本
        transform (callable):用来转换batch samples的 function/transform
    """
    # 以tensor形式将单通道标签转换为RGB
    # 1. torch.unbind删除"标签"的0维，并返回沿该维度的所有slice的元组
    # 2. transform作用于每个slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)


def imshow_batch(images, labels):
    """显示两个图像网格,顶部网格显示"image"和底部网格"label"

    Args:
        images ("Tensor"): a 4D mini-batch tensor,shape 是 (B, C, H, W)
    - labels ("Tensor"): a 4D mini-batch tensor, shape 是 (B, C, H, W)

    """
    # 使用image和label制作网格并将其转换为numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, args):
    """指定文件夹和名字存储模型文件

    Args:
        model ("nn.Module"): model
        optimizer ("torch.optim"): 存储的优化器.
        epoch ("int"): The current epoch for the model.模型的当前epoch
        miou ("float"): 由model获得的mean IoU
        args ("ArgumentParser"): ArgumentParser的一个实例，包含用于训练model的参数。 参数被写入名为"args.name"_args.txt"的"args.save_dir"中的文本文件。
    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """使用指定的name.save将model保存在指定的目录中，load checkpoint

    Args:
        model ("nn.Module"): 存储的模型状态将复制到此模型实例
        optimizer ("torch.optim"): 存储的optimizer程序状态将复制到此optimizer程序实例.
        folder_dir ("string"): 保存的model state 所在文件夹的路径。
        filename ("string"): 模型文件名

    Returns:
        返回从checkpoint中加载的epoch, mean IoU, "model", 和 "optimizer" 

    """
    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # 创建文件夹来存储model和information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)
    print("加载已经存储的模型")
    # 加载已经存储的模型参数到model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou
