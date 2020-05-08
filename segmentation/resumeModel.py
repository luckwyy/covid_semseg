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

from CTData_3label import get_data_dcm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Fill in the relevant parameters here
# author: wy 20200509
base_model = 'deeplabv3plus_resnet50'
model_name = './checkpoints/deeplabv3plus_resnet50_dice_avg_0.824_dice_bl_0.795_dice_sb_0.851_dice_st_0.826__98500.pkl'


def get_argparser():
    parser = argparse.ArgumentParser()

    # Deeplab Options
    parser.add_argument("--model", type=str, default=base_model,
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--total_itrs", type=int, default=100000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=10,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')

    parser.add_argument("--ckpt", default=model_name, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

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

    iou_bl = 0
    iou_st = 0
    iou_sb = 0

    count = 0

    bl_tp = 0
    bl_tn = 0
    bl_fp = 0
    bl_fn = 0

    sb_tp = 0
    sb_tn = 0
    sb_fp = 0
    sb_fn = 0

    st_tp = 0
    st_tn = 0
    st_fp = 0
    st_fn = 0
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            count += 1
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

            di = get_IOU(SR=out_sigmoid[:, 0, :, :], GT=labels[:, 0, :, :])
            iou_bl += di
            di = get_IOU(SR=out_sigmoid[:, 1, :, :], GT=labels[:, 1, :, :])
            iou_sb += di
            di = get_IOU(SR=out_sigmoid[:, 2, :, :], GT=labels[:, 2, :, :])
            iou_st += di


            tp, tn, fp, fn = get_TPNFPN(SR=out_sigmoid[:, 0, :, :], GT=labels[:, 0, :, :])
            bl_tp += tp
            bl_tn += tn
            bl_fp += fp
            bl_fn += fn
            tp, tn, fp, fn = get_TPNFPN(SR=out_sigmoid[:, 1, :, :], GT=labels[:, 1, :, :])
            sb_tp += tp
            sb_tn += tn
            sb_fp += fp
            sb_fn += fn
            tp, tn, fp, fn = get_TPNFPN(SR=out_sigmoid[:, 2, :, :], GT=labels[:, 2, :, :])
            st_tp += tp
            st_tn += tn
            st_fp += fp
            st_fn += fn
            if count % 50 == 0:
                print(count, iou_bl / count)

        acc = acc / loader.dataset.__len__()
        dice_bl = dice_bl / loader.dataset.__len__()
        dice_sb = dice_sb / loader.dataset.__len__()
        dice_st = dice_st / loader.dataset.__len__()
        dice_loss = dice_loss / loader.dataset.__len__()

        iou_bl = iou_bl / loader.dataset.__len__()
        iou_sb = iou_sb / loader.dataset.__len__()
        iou_st = iou_st / loader.dataset.__len__()

        path = base_model+'.csv'

        content = 'dice_bl: {}, dice_sb: {}, dice_st: {}, iou_bl: {}, iou_sb: {}, iou_st: {}, bl_tp: {}' \
                  ', bl_tn{}, bl_fp: {}, bl_fn: {}, ' \
                  'sb_tp: {}, sb_tn: {}, sb_fp: {}, sb_fn: {},' \
                  'st_tp: {}, st_tn: {}, st_fp: {}, st_fn: {}'.format(
            dice_bl.item(), dice_sb.item(), dice_st.item(), iou_bl.item(), iou_sb.item(), iou_st.item(),
            bl_tp, bl_tn, bl_fp, bl_fn,
            sb_tp, sb_tn, sb_fp, sb_fn,
            st_tp, st_tn, st_fp, st_fn
        )

        print(content)

        bl_acc = (bl_tp + bl_tn) / (bl_tp + bl_tn + bl_fp + bl_fn)
        bl_r = bl_tp / (bl_tp + bl_fn)
        bl_p = bl_tp / (bl_tp + bl_fp)
        bl_f1 = 2 * bl_p * bl_r / (bl_p + bl_r)

        sb_acc = (sb_tp + sb_tn) / (sb_tp + sb_tn + sb_fp + sb_fn)
        sb_r = sb_tp / (sb_tp + sb_fn)
        sb_p = sb_tp / (sb_tp + sb_fp)
        sb_f1 = 2 * sb_p * sb_r / (sb_p + sb_r)

        st_acc = (st_tp + st_tn) / (st_tp + st_tn + st_fp + st_fn)
        st_r = st_tp / (st_tp + st_fn)
        st_p = st_tp / (st_tp + st_fp + 0.001)
        st_f1 = 2 * st_p * st_r / (st_p + st_r + 0.001)

        avg_acc = (bl_acc+sb_acc+st_acc) / 3
        avg_r = (bl_r+sb_r+st_r) / 3
        avg_p = (bl_p+sb_p+st_p) / 3
        avg_f1 = (bl_f1+sb_f1+st_f1) / 3

        content_3 = "{},{},{},{}".format(
            iou_bl, iou_sb, iou_st, (iou_bl + iou_sb+iou_st) / 3
        )
        content_2 = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
            bl_acc,sb_acc,st_acc,avg_acc,
            bl_r,sb_r,st_r,avg_r,
            bl_p,sb_p,st_p,avg_p,
            bl_f1,sb_f1,st_f1,avg_f1,
        )
        print(content_2)

        with open(path, mode='w', encoding='utf-8') as f:
            f.write(content+'\n'+content_3+'\n'+content_2)


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
    if base_model != 'unet':
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


if __name__ == '__main__':
    main()
