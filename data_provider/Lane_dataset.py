import os
from collections import OrderedDict
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image


class Lane_dataset(data.Dataset):
    """
    """
    # Default encoding for pixel value, class name, and class color

    color_encoding = OrderedDict([
                 ('unlabeled', (0,0,0)),
                 ('1', (180, 130, 70)),
                 ('2', (142, 0, 0)),
                 ('3', (153, 153, 153)),
                 ('4', (156, 102, 102)),
                 ('5', (128, 64, 128)),
                 ('6', (153, 153, 190)),
                 ('7', (230, 0, 0)),
                 ('8', (0, 128, 255))])

    color_encoding_bgr = OrderedDict([
                 ('0', [[255, 255, 255], [0,0,0], [153,153,0]]),
                 ('1', [[180, 130, 70],[60, 20, 220], [128, 0, 128], [0, 0, 255],[60, 0, 0], [100, 60, 0]]),
                 ('2', [[142, 0, 0], [32, 11, 119], [232, 35, 224], [160, 0, 0]]),
                 ('3', [[153, 153, 153], [0, 220, 220], [30, 170, 250]]),
                 ('4', [[156, 102, 102], [0, 0, 128]]),
                 ('5', [[128, 64, 128], [170, 232, 238]]),
                 ('6', [[153, 153, 190]]),
                 ('7',
                  [[230, 0, 0],
                   [0, 128, 128],
                   [160, 78, 128],
                   [100, 100, 150],
                   [0, 165, 255],
                   [180, 165, 180],
                   [35, 142, 107],
                   [229, 255, 201],
                   [255, 191, 0],
                   [51, 255, 51],
                   [114, 128, 250],
                   [0, 255, 175]]),
                 ('8',
                  [[0, 128, 255],
                   [255, 255, 0],
                   [190, 132, 178],
                   [64, 128, 128],
                   [204, 0, 102]])])

    def __init__(self,
                 dataset_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 #loader=utils.pil_loader
                 ):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        # self.loader = loader
        
        # 全部数据
        self.all_data = []
        self.all_labels = []
        
        # 训练数据
        self.train_data = []
        self.train_labels = []
        
        # 验证数据
        self.val_data = []
        self.val_labels = []
        
        # 测试数据
        self.test_data = []
        self.test_labels = []
        
        # 获取所有的训练，测试图片的路径
        self.get_datapath(self.dataset_dir)

    def get_datapath(self,path):
        files= os.listdir(path) # 得到文件夹下面的所有文件的名称
        for dirs in files:
            if "train" in dirs:
                self.all_data = ["{}/{}/{}".format(path,dirs,img_name) for img_name in os.listdir("{}/{}".format(path,dirs))]
            else:
                self.all_labels = ["{}/{}/{}".format(path,dirs,img_name) for img_name in os.listdir("{}/{}".format(path,dirs))]
                
        data_length = len(self.all_labels)

        self.train_data = self.all_data[0:int(data_length * 0.7)]
        self.train_labels = self.all_labels[0:int(data_length * 0.7)]
        
        self.val_data = self.all_data[int(data_length * 0.7):int(data_length * 0.9)]
        self.val_labels = self.all_labels[int(data_length * 0.7):int(data_length * 0.9)]
        
        self.test_data = self.all_data[int(data_length * 0.9):-1]
        self.test_labels = self.all_labels[int(data_length * 0.9):-1]
    
        
        # self.train_data = self.all_data[0:int(data_length * 0.7)]
        # self.train_labels = self.all_labels[0:int(data_length * 0.7)]
        
        # self.val_data = self.all_data[int(data_length * 0.7):-1]
        # self.val_labels = self.all_labels[int(data_length * 0.7):-1]
        
    
    def __getitem__(self, index):
        """
        Args:
            index ("int"): dataset中的索引

        Returns:
            元组, "PIL.Image" (image, label) label 是图片中的ground-truth

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img = Image.open(data_path)
        label_ = Image.open(label_path)
        label = Image.fromarray(np.array(label_)[:,:,0])

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")