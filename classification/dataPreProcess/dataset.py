import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import os.path as op
import SimpleITK as sitk
from myutils import txtUtils

import matplotlib.pyplot as plt

class subDataset(Dataset):
    def __init__(self, files, dcms_path, masks_path):
        self.files = files
        self.dcms_path = dcms_path

        self.bl_masks_path = masks_path+'bl/'
        self.sb_masks_path = masks_path+'sb/'
        self.st_masks_path = masks_path+'st/'

        self.bl_path_files = [op.splitext(i)[0] for i in os.listdir(masks_path+'bl/')]
        self.sb_path_files = [op.splitext(i)[0] for i in os.listdir(masks_path+'sb/')]
        self.st_path_files = [op.splitext(i)[0] for i in os.listdir(masks_path+'st/')]


    def __getitem__(self, item):

        dcm_path = self.dcms_path+self.files[item]+'.dcm'
        # png_path = self.masks_path+self.files[item]+'.dcm'

        dcm_nd = self.dcmPreProcess(dcm_path = dcm_path)
        png_nd = self.pngPreProcess(png_name = self.files[item])

        return [torch.from_numpy(dcm_nd), torch.from_numpy(png_nd), self.files[item]]

    def dcmPreProcess(self, dcm_path):
        dcm = sitk.ReadImage(dcm_path, sitk.sitkInt16)
        dcm_nd = sitk.GetArrayFromImage(dcm)
        # 如果是666尺寸 裁剪到512
        if dcm_nd.shape[1] == 666:
            dcm_nd = dcm_nd[:, 77:589, :]
        # 归一化
        new_dcm_nd = (dcm_nd + 975.6) / 1161.2
        data = np.concatenate([new_dcm_nd, new_dcm_nd, new_dcm_nd])

        return data

    def pngPreProcess(self, png_name):
        # label = torch.zeros([3]).float()
        label = np.zeros((3))
        if png_name in self.bl_path_files:
            label[0] = 1.
        if png_name in self.sb_path_files:
            label[1] = 1.
        if png_name in self.st_path_files:
            label[2] = 1.

        return label


    def __len__(self):
        return len(self.files)


def main():

    files = txtUtils.getTxtContentList('D:/paperDatasets/label2mix-3/train_list.txt')
    train_set = subDataset(files = files, dcms_path='D:/paperDatasets/data2mix/', masks_path='D:/paperDatasets/label2mix-3/')

    train_loader = DataLoader(train_set, batch_size=4, num_workers=0, shuffle=True, drop_last=True)
    for batch in train_loader:
        print(batch[0].shape, batch[1], batch[1].shape)


if __name__ == '__main__':
    main()