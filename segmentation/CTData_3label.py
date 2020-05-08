import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import torchvision
import pydicom
import SimpleITK as sitk


class CTData_3label(Dataset):
    def __init__(self, img_h=512, img_w=512, data_list=[], iscrop=True):
        self.img_h = img_h
        self.img_w = img_w
        self.data_list = data_list

        self.iscrop = iscrop

        self.file_path = "D:/paperDatasets/"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        file_name, data_name = self.data_list[index].split('dcm')
        path = self.file_path+"data2mix/" + file_name + "dcm" + data_name + ".dcm"
        ct = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(ct)
        img_h = image.shape[1]
        img_w = image.shape[2]
        min_img = image.min()
        max_img = image.max()
        ranges = max_img - min_img
        newimg1 = (image - min_img) / ranges
        newimg1 = (newimg1 - 0.5) * 2
        image = np.concatenate([newimg1, newimg1, newimg1])
        label_name = self.data_list[index]
        path = self.file_path+"label2mix-3/bl/" + label_name + ".dcm"
        if os.path.isfile(path):
            ct = sitk.ReadImage(path)
            label_bl = sitk.GetArrayFromImage(ct)

        else:
            label_bl = np.zeros([1, img_h, img_w])

        path = self.file_path+"label2mix-3/sb/" + label_name + ".dcm"
        if os.path.isfile(path):
            ct = sitk.ReadImage(path)
            label_sb = sitk.GetArrayFromImage(ct)

        else:
            label_sb = np.zeros([1, img_h, img_w])

        path = self.file_path+"label2mix-3/st/" + label_name + ".dcm"
        if os.path.isfile(path):
            ct = sitk.ReadImage(path)
            label_st = sitk.GetArrayFromImage(ct)

        else:
            label_st = np.zeros([1, img_h, img_w])

        label_numpy = np.concatenate([label_bl, label_sb, label_st])
        if self.iscrop:
            image, label_numpy = CenterCrop(img=image, lbl=label_numpy, size_in=(img_w, img_h),
                                            size_out=(self.img_w, self.img_h))

        image = torch.from_numpy(image)
        label = torch.from_numpy(label_numpy).float()

        return image, label


def RandomCrop(img, lbl, size_in, size_out):
    w, h = size_in
    tw, th = size_out

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return img[:, i:i + th, j:j + tw], lbl[:, i:i + th, j:j + tw]


def CenterCrop(img, lbl, size_in, size_out):
    w, h = size_in
    tw, th = size_out

    i = int((h - th) / 2)
    j = int((w - tw) / 2)
    return img[:, i:i + th, j:j + tw], lbl[:, i:i + th, j:j + tw]


def getTxtContentList(path):
    with open(path, mode='r', encoding='utf-8') as f:
        content = f.read().splitlines()
    return content


def get_data_dcm(img_h=400, img_w=400, path='D:/paperDatasets/label2mix-3', iscrop=True):
    train_list = getTxtContentList(path=path + "/train_list.txt")
    test_list = getTxtContentList(path=path + "/test_list.txt")

    print("train data number",len(train_list))
    print("test data number", len(test_list))
    train_data = CTData_3label(img_h=img_h, img_w=img_w, data_list=train_list, iscrop=iscrop)
    test_data = CTData_3label(img_h=img_h, img_w=img_w, data_list=test_list, iscrop=iscrop)
    return train_data, test_data


if __name__ == '__main__':
    # data_list = os.listdir(  "./10.dcm")
    # ct = get_data_unlabel()
    # img = ct.__getitem__(10)
    train_data, test_data = get_data_dcm(img_h=512, img_w=512, iscrop=True)
    for i in range(20):
        data, label = train_data.__getitem__(i)
        print(data.size())
        print(label.size())
    train_loader = DataLoader(
        train_data, batch_size=8, shuffle=True, num_workers=0)
    train_iter = iter(train_loader)
    train_iter.__next__()
