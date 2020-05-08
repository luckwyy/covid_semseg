import network
import utils
import os
import random
import argparse
import numpy as np
from evaluation import *
from torch.utils import data
from unet import UNet

import torch
import torch.nn as nn
import torchvision

import txtUtils

from CTData_3label import get_data_dcm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# resume_model_name = './checkpoints/deeplabv3plus_resnet101_dice_avg_0.444_dice_bl_0.338_dice_sb_0.487_dice_st_0.508__500.pkl'

def get_argparser():
    parser = argparse.ArgumentParser()

    # Deeplab Options
    parser.add_argument("--model", type=str, default='unet',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=100000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=3,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')

    parser.add_argument("--ckpt", default="", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=True)

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    return parser


def validate2(model, loader, device, itrs, lr, criterion_dice):
    """Do validation and return specified samples"""

    dice_bl = 0
    dice_st = 0
    dice_sb = 0
    acc = 0
    dice_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            # labels = labels.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(images)
            out_sigmoid = torch.sigmoid(outputs)
            dice_loss += criterion_dice(out_sigmoid, labels)
            acc += get_accuracy(SR=out_sigmoid, GT=labels)
            # label_numpy = np.concatenate([label_bl, label_sb, label_st])
            di = get_Dice(SR=out_sigmoid[:, 0, :, :], GT=labels[:, 0, :, :])
            dice_bl += di
            di = get_Dice(SR=out_sigmoid[:, 1, :, :], GT=labels[:, 1, :, :])
            dice_sb += di
            di = get_Dice(SR=out_sigmoid[:, 2, :, :], GT=labels[:, 2, :, :])
            dice_st += di

            # Select some test data to present
            if i < 5:
                out = torch.where(outputs > 0, torch.full_like(outputs, 1), torch.full_like(outputs, 0))
                # out_cat = torch.where(outputs_all > 0, torch.full_like(outputs_all, 1), torch.full_like(outputs_all, 0))
                images = (images.data.cpu() + 1) / 2
                h, w = images.size(2), images.size(3)
                out1 = torch.zeros([3, h, w]) + out.data.cpu()[:, 0:1, :, :]
                out2 = torch.zeros([3, h, w]) + out.data.cpu()[:, 1:2, :, :]
                out3 = torch.zeros([3, h, w]) + out.data.cpu()[:, 2:3, :, :]

                labels1 = torch.zeros([3, h, w]) + labels.data.cpu()[:, 0:1, :, :]
                labels2 = torch.zeros([3, h, w]) + labels.data.cpu()[:, 1:2, :, :]
                labels3 = torch.zeros([3, h, w]) + labels.data.cpu()[:, 2:3, :, :]
                save_tensor = torch.cat([images, out1, out2, out3, labels1, labels2, labels3], 0)

                torchvision.utils.save_image(save_tensor, os.path.join(
                    "./saved/itr_" + str(itrs) + "_i_" + str(i) + "_lr_" + str(lr) + ".png"))
        acc = acc / loader.dataset.__len__()
        dice_bl = dice_bl / loader.dataset.__len__()
        dice_sb = dice_sb / loader.dataset.__len__()
        dice_st = dice_st / loader.dataset.__len__()
        dice_loss = dice_loss / loader.dataset.__len__()
        print("dice_bl:", dice_bl.item(), "dice_sb:", dice_sb.item(), "dice_st:", dice_st.item(), " acc:", acc,
              " dice_loss", dice_loss.item(), )
        return dice_bl.item(), dice_sb.item(), dice_st.item(), acc


def main():
    opts = get_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    img_h = 512
    img_w = 512

    torch.cuda.empty_cache()
    train_data, test_data = get_data_dcm(img_h=img_h, img_w=img_w, iscrop=True)
    train_loader = data.DataLoader(
        train_data, batch_size=opts.batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(
        test_data, batch_size=opts.val_batch_size, shuffle=False, num_workers=0)

    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    if opts.model != 'unet':
        opts.num_classes = 3
        model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)

        # Set up optimizer
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        opts.num_classes = 3
        model = UNet(n_channels=3, n_classes=3, bilinear=True)

        # Set up optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)

    criterion_bce = nn.BCELoss(reduction='mean')
    criterion_dice = MulticlassDiceLoss()

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.test_only:
        # model.load_state_dict()
        model.eval()
        dice_bl, dice_sb, dice_st, acc = validate2(model=model, loader=val_loader, device=device, itrs=cur_itrs,
                                                   lr=scheduler.get_lr()[-1],
                                                   criterion_dice=criterion_dice)
        # save_ckpt("./checkpoints/CT_" + opts.model + "_" + str(round(dice, 3)) + "__" + str(cur_itrs) + ".pkl")
        print("dice值：", dice_bl)
        return
    best_dice_bl = 0
    best_dice_sb = 0
    best_dice_st = 0
    best_dice_avg = 0
    interval_loss = 0
    train_iter = iter(train_loader)

    txt_path = './train_info.txt'
    # txtUtils.clearTxt(txt_path)

    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        try:
            images, labels = train_iter.__next__()
        except:
            train_iter = iter(train_loader)
            images, labels = train_iter.__next__()

        cur_itrs += 1
        # print(images.size())
        # print(labels.size())
        images = images.to(device, dtype=torch.float32)
        # labels = labels.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float32)
        # print(images.size())

        outputs = model(images)
        outputs_ = torch.sigmoid(outputs)

        loss = criterion_bce(outputs_, labels) + criterion_dice(outputs_, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        np_loss = loss.item()
        interval_loss += np_loss

        if (cur_itrs) % 50 == 0:
            interval_loss = interval_loss / 50
            cur_epochs = int(cur_itrs/train_loader.dataset.__len__())
            print("Epoch %d, Itrs %d/%d, Loss=%f" %
                  (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))

            content = ("Epoch {}, Itrs {}/{}, Loss={}").format(cur_epochs, cur_itrs, opts.total_itrs, interval_loss)
            txtUtils.writeInfoToTxt(file_path=txt_path,content=content, is_add_time=True)
            interval_loss = 0.0
        # opts.val_interval=5
        if (cur_itrs) % 500 == 0:
            print("validation... lr:", scheduler.get_lr())
            content = ("validation... lr:{}").format(scheduler.get_lr())
            txtUtils.writeInfoToTxt(file_path=txt_path,content=content, is_add_time=True)
            # print(outputs)
            dice_bl, dice_sb, dice_st, acc = validate2(model=model, loader=val_loader, device=device, itrs=cur_itrs,
                                                       lr=scheduler.get_lr()[-1], criterion_dice=criterion_dice)
            dice_avg = (dice_bl + dice_sb + dice_st) / 3

            content = ("dice_bl:{}, dice_sb:{}, dice_st:{}, acc:{}, dice_avg:{}").format(dice_bl, dice_sb, dice_st, acc, dice_avg)
            txtUtils.writeInfoToTxt(file_path=txt_path,content=content, is_add_time=True)

            if best_dice_avg < dice_avg:
                best_dice_avg = dice_avg
                save_ckpt("./checkpoints/" + opts.model + "_dice_avg_" + str(round(best_dice_avg, 3))
                          + "_dice_bl_" + str(round(dice_bl, 3))
                          + "_dice_sb_" + str(round(dice_sb, 3))
                          + "_dice_st_" + str(round(dice_st, 3))
                          + "__" + str(cur_itrs) + ".pkl")
                print("best avg dice：", best_dice_avg)
                content = ("best avg dice: {}").format(best_dice_avg)
                txtUtils.writeInfoToTxt(file_path=txt_path,content=content, is_add_time=True)

            if best_dice_bl < dice_bl:
                best_dice_bl = dice_bl
                content = ("best bl dice: {}").format(best_dice_bl)
                txtUtils.writeInfoToTxt(file_path=txt_path,content=content, is_add_time=True)
            if best_dice_sb < dice_sb:
                best_dice_sb = dice_sb
                content = ("best sb dice: {}").format(best_dice_sb)
                txtUtils.writeInfoToTxt(file_path=txt_path,content=content, is_add_time=True)
            if best_dice_st < dice_st:
                best_dice_st = dice_st
                content = ("best st dice: {}").format(best_dice_st)
                txtUtils.writeInfoToTxt(file_path=txt_path,content=content, is_add_time=True)

        scheduler.step()

        if cur_itrs >= opts.total_itrs:
            return


if __name__ == '__main__':
    main()
