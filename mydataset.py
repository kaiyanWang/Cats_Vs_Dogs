import torch.utils.data
import numpy as np
import os, random, glob
from torchvision import transforms
from PIL import Image

# 定义数据集读取类
class DogCatDataSet(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform

        dog_dir = os.path.join(img_dir, "dog")
        cat_dir = os.path.join(img_dir, "cat")
        imgsLib = []
        imgsLib.extend(glob.glob(os.path.join(dog_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(cat_dir, "*.jpg")))
        random.shuffle(imgsLib)  # 打乱数据集
        self.imgsLib = imgsLib

    # 作为迭代器必须要有的
    def __getitem__(self, index):
        img_path = self.imgsLib[index]

        label = 1 if 'dog' in img_path.split('/')[-1] else 0 #狗的label设为1，猫的设为0

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgsLib)