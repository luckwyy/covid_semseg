from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
from dataPreProcess.dataset import subDataset
from torch.utils.data import DataLoader
from myutils import txtUtils
from torch.optim.lr_scheduler import StepLR
import model

# from model list select one model:
#     resnet18
#     resnet50
#     resnet101
#     densenet121
#     googlenet

os.environ["CUDA_VISIBLE_DIVICES"] = "0"

device = torch.device("cuda")

###################################
model_name = 'resnet101'
batch_size = 6
epochs = 50
load_exist_model = './checkpoints/'+''
dcms_path = 'D:/paperDatasets/data2mix/'
masks_path = 'D:/paperDatasets/label2mix-3/'
train_list_path = 'D:/paperDatasets/label2mix-3/train_list.txt'
test_list_path = 'D:/paperDatasets/label2mix-3/test_list.txt'
out_dir = './res/'
###################################


model = model.covid_net(name=model_name).to(device)
if load_exist_model != './checkpoints/'+'':
    model.load_state_dict(torch.load(load_exist_model))
criteon = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

files = txtUtils.getTxtContentList(train_list_path)
train_set = subDataset(files=files, dcms_path=dcms_path, masks_path=masks_path)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)

files = txtUtils.getTxtContentList(test_list_path)
test_set = subDataset(files=files, dcms_path=dcms_path, masks_path=masks_path)
test_loader = DataLoader(test_set, batch_size=1)



txt_path = out_dir+model_name+'-train_info.txt'
txtUtils.clearTxt(txt_path)
step = 0
csv_path = out_dir+model_name+'-test_info.csv'
with open(csv_path, mode='w', encoding='utf-8') as f:
    pass
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        data = batch[0].float().to(device)
        label = batch[1].float().to(device)

        pred = model(data)

        optimizer.zero_grad()
        loss = criteon(pred, label)
        loss.backward()
        optimizer.step()

        step += 1
        content = "epoch: {}, step: {}, loss: {}".format(epoch, step, loss.cpu().item())
        print(content)
        if step % 100 == 0:
            txtUtils.writeInfoToTxt(file_path=txt_path, content=content, is_add_time=True)

    tp = 0

    fn = 0

    fp = 0

    tn = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].float().to(device)
            label = batch[1].float().to(device)
            pred = model(data)
            loss = criteon(pred, label)

            loss = loss.cpu().numpy()
            pred = pred.cpu().numpy()[0]
            label = label.cpu().numpy()[0]
            content = "epoch: {}, step: {}, file_name: {}, loss: {}, pred: {}, label: {}".format(epoch, step, batch[2][0], loss, pred, label)
            print(content)
            txtUtils.writeInfoToTxt(file_path=txt_path, content=content, is_add_time=True)

            with open(csv_path, mode='a', encoding='utf-8') as f:
                content = "{},{},{},{},{},{},{},{}".format("epoch-"+str(epoch), batch[2][0], pred[0], pred[1], pred[2], label[0], label[1], label[2])
                f.write(content+'\n')

            # bl
            if label[0] == 1:
                if pred[0] > 0.5:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[0] > 0.5:
                    fp += 1
                else:
                    tn += 1

            if label[1] == 1:
                if pred[1] > 0.5:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[1] > 0.5:
                    fp += 1
                else:
                    tn += 1

            if label[2] == 1:
                if pred[2] > 0.5:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[2] > 0.5:
                    fp += 1
                else:
                    tn += 1

    # acc
    Acc = (tp + tn) / (tp + tn + fp + fn)
    # r
    Precision = tp / (tp + fp)
    # p
    Recall = tp / (tp + fn)


    content = "epoch: {}, acc: {}, r: {}, p: {}".format(epoch, Acc, Precision, Recall)
    print(content)
    txtUtils.writeInfoToTxt(file_path=txt_path, content=content, is_add_time=True)

    lr_scheduler.step()

    torch.save(model.state_dict(), "./checkpoints/"+model_name+"-checkpoint-"+str(epoch)+'.pth')