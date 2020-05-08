import torch


# SR : Segmentation Result
# GT : Ground Truth
import numpy as np

def get_accuracy(SR, GT, threshold=0.5):
    # SR = SR > threshold
    SR = torch.where(SR > threshold, torch.full_like(SR, 1), torch.full_like(SR, 0))
    # GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    # tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    tensor_size =  SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc

def get_Dice(SR, GT, threshold=0.5):
    # logits = logits > threshold
    SR = torch.where(SR > threshold, torch.full_like(SR, 1), torch.full_like(SR, 0))
    inter = torch.sum(SR * GT)
    union = torch.sum(SR) + torch.sum(GT)
    dice = (2. * inter + 0.001) / (union + 0.001)
    return dice

def get_IOU(SR, GT, threshold=0.5):
    iou = 0
    SR = torch.where(SR > threshold, torch.full_like(SR, 1), torch.full_like(SR, 0))
    inter = torch.sum(SR * GT)
    union = torch.sum(SR) + torch.sum(GT) - inter
    iou = (inter + 0.001) / (union + 0.001)

    return iou

#
def get_TPNFPN(SR, GT, threshold=0.5):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    SR = torch.where(SR > threshold, torch.full_like(SR, 1), torch.full_like(SR, 0))
    #
    label = 0
    pred = 0
    if GT.max() >= 1:
        label = 1
    if SR.max() >= 1:
        pred = 1

    if label == 1:
        if pred == 1:
            tp = 1
        else:
            fn = 1
    else:
        if pred == 0:
            tn = 1
        else:
            fp = 1
    # print(tp, tn, fp, fn)
    return [tp, tn, fp, fn]


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2.)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 0.01

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss =  (2 *intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = -torch.log(loss.sum() / N)

        return loss


class MulticlassDiceLoss(nn.Module):


    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()
        self.dice = DiceLoss()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes


        totalLoss = 0

        for i in range(C):
            diceLoss = self.dice(input[:, i,:,:], target[:, i,:,:])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss
        totalLoss = totalLoss/C
        return totalLoss


def main():

    masks = torch.tensor([[1,0,0], [1,1,0], [1,0,1]])
    pred = torch.tensor([[0.6, 0.5, 0.5], [0.6, 0.6, 0.5], [0.6, 0.5, 0.6]])

    iou = get_IOU(pred, masks)
    print('iou=', iou)


if __name__ == '__main__':
    main()