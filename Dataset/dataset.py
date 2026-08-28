import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
def add_gaussian_noise(image, mean=0., std=1.):
    # 将Tensor转换为NumPy数组
    np_image = image.numpy().astype(np.float32)
    # 添加高斯噪声
    np_image = np_image + np.random.normal(mean, std, np_image.shape)
    # 确保值在0-1范围内
    np_image = np.clip(np_image, 0., 1.)
    # 将NumPy数组转换回Tensor
    return torch.from_numpy(np_image).float()

def show_image_and_label(img):

    # 将Tensor转换为NumPy数组，因为matplotlib需要NumPy数组或PIL图像
    img = img.numpy().transpose((1, 2, 0))  # 转换为HxWxC

    # 对图像进行反归一化处理（如果进行了归一化）
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img = std * img + mean
    img = np.clip(img, 0, 1)  # 限制图像数组的值在0和1之间
    return img

def random_shape_with_mask(image,label = None,size = 224):
    shape = transforms.Compose([
        transforms.RandomCrop(size=size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ])
    print(image.shape)
    print(label.shape)
    if label is None:
        image = shape(image)
        return image
    else:
        combine = torch.concat((image,label),dim=0)
        combine = shape(combine)
        image = combine[:3]
        label = combine[3:]
        return image,label

def apply_transform_train(image, label = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.GaussianBlur(kernel_size=5,sigma=(0.1,2.0)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # image = add_gaussian_noise(image) # 处理过拟合才加的
    label = transforms.ToTensor()(label) if label is not None else None
    if label is not None:
        image,label = random_shape_with_mask(image,label,256)
        return image,label
    else:
        image = random_shape_with_mask(image)
        return image

def apply_transform_test(image, label = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    label = transforms.ToTensor()(label) if label is not None else None
    if label is not None:
        image,label = random_shape_with_mask(image,label,256)
        return image,label
    else:
        image = random_shape_with_mask(image)
        return image
# def change_label(label):
#     label = np.array(label)
#     label_bg = 255 - label
#     label_front = label
#     label_bg = np.expand_dims(label_bg,axis=-1)
#     label_front = np.expand_dims(label_front,axis=-1)
#     label = np.concatenate((label_bg,label_front),axis=2)
#     label = Image.fromarray(label)
#     return label


class PathologyData(Dataset):
    def __init__(self, dataset_folder, dataset_label_folder=None,train = True,apply_transform = True):
        self.dataset_folder = sorted([
            os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
            if any(f.endswith(ext) for ext in ['.png', '.tif','bmp'])
        ], key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        self.apply_transform = apply_transform
        self.dataset_label_folder = dataset_label_folder
        if self.dataset_label_folder is not None:
            self.dataset_label_folder = sorted([
                os.path.join(dataset_label_folder, f) for f in os.listdir(dataset_label_folder)
                if any(f.endswith(ext) for ext in ['.png', '.tif','bmp'])
            ],key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        print(self.dataset_folder)
        print(self.dataset_label_folder)
        self.train = train
    def __len__(self):
        return len(self.dataset_folder)

    def __getitem__(self, item):
        img_path = self.dataset_folder[item]
        img = Image.open(img_path).convert('RGB')  # 确保是RGB格式

        label_path = self.dataset_label_folder[item] if self.dataset_label_folder is not None else None
        label = Image.open(label_path).convert('L') if label_path else None

        if self.apply_transform:
            img, label = apply_transform_train(img, label) if self.train else apply_transform_test(img,label)
        else:
            img, label = img, label

        if label is None:
            return img
        else:
            return img, label



class glas_classify_Dataset(Dataset):
    def __init__(self,csv_path,binary = True,train = True):
        super().__init__()
        self.csv_path = csv_path
        csv = pd.read_csv(self.csv_path)
        self.train = train
        # 判断是否是用于训练
        if train:
            csv = csv[csv['name'].str.contains('train')]
        else:
            csv = csv[csv['name'].str.contains('test')]

        # 判断label是要用哪种标准进行分类
        if binary:
            self.label_csv = csv.iloc[:,2] # 2分类
        else:
            self.label_csv = csv.iloc[:,3] # 5分类

        self.image_csv = csv.iloc[:,0]

        self.image_list = self.image_csv.tolist()
        self.label_list = self.label_csv.tolist()


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,item):
        if self.train:
            image = Image.open(self.image_list[item]).convert('RGB')
            label = self.label_list[item]
            label = torch.tensor(label)

            image = apply_transform_train(image)
            return image,label
        else:
            image = Image.open(self.image_list[item]).convert('RGB')
            label = self.label_list[item]
            label = torch.tensor(label)

            image = apply_transform_test(image)
            return image,label

if __name__ == '__main__':
    # dataset = PathologyData(dataset_folder='Dataset/glas/train/image',dataset_label_folder='Dataset/glas/train/mask')
    # image,label = dataset[0]
    # print(image.shape)
    # print(label.shape)
    # print(torch.max(image))
    # print(torch.min(image))
    # print(torch.unique(label))
    dataset = glas_classify_Dataset(csv_path='Dataset/glas/Grade_modified.csv')
    image,label = dataset[0]
    print(image.shape)
    print(label)
