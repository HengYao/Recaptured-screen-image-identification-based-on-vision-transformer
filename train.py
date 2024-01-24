import os
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from Focal_Loss import FocalLoss
import torch.nn as nn
from net import MyNetwork
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False
LR = 0.001
weight_decay = 0.000000000000
EPOCH = 100
BATCHSIZE = 128
print('epoch=%d'%(EPOCH),'batchsize=%d'%(BATCHSIZE),'LR=%.10f'%(LR))

# 预处理
transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.ImageFolder('',transform=transform)  # train dataset path
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True,drop_last=True, num_workers=0)
test_dataset = torchvision.datasets.ImageFolder('', transform=transform) # test dataset path
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=True,num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyNetwork().to(device)

weight = 0.6
loss_fn1 = nn.CrossEntropyLoss()
loss_fn = FocalLoss()

optimizer = torch.optim.Adam(params=model.parameters(),lr=LR, weight_decay=weight_decay)
lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)


def get_weight(model):
    weight_list = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight = (name, param)
            weight_list.append(weight)
    return weight_list

def regularization_loss(weight_list, weight_decay):
    reg_loss = 0
    for name, w in weight_list:
        l2_reg = torch.pow(w,2)
        reg_loss = reg_loss + l2_reg
    regularzation_loss = weight_decay * reg_loss
    return regularzation_loss


def train(dataloder,model,loss_fn,optimizer):
    loss,current,n,regularzation_loss = 0.0,0.0,0,0
    for batch,(x,y) in enumerate(dataloder):
        # l2_loss = []
        image,y = x.to(device),y.to(device)
        output = model(image)
        current_loss = loss_fn1(output,y) #+ weight_decay * regularzation_loss
        _,pred = torch.max(output,1)
        cur_acc = torch.sum(y == pred)/output.shape[0]

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()

        current += cur_acc.item()
        n = n+1

    train_loss = loss/n
    train_acc = current/n

    print('train_loss:'+ str(train_loss))
    print('train_acc:'+ str(train_acc))
    return train_loss,train_acc


def val(dataloder, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloder):
            image, y = x.to(device), y.to(device)
            output = model(image)
            current_loss = loss_fn(output, y)
            _, pred = torch.max(output, 1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += current_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val_loss:' + str(val_loss))
    print('val_acc:' + str(val_acc))
    return val_loss, val_acc
def matplot_loss(train_loss,val_loss):
    plt.plot(train_loss,label = 'train_loss')
    plt.plot(val_loss,label = 'val_loss')
    plt.legend(loc = 'best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.title('训练集和验证集loss值对比图')
    plt.show()

def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy rate ')
    # plt.title('训练集和验证集acc值对比图')
    plt.show()

loss_train = []
acc_train = []
loss_val = []
acc_val = []

min_acc = 0.8467

for t in range(EPOCH):
    lr_scheduler.step()
    print(f"epoch{t+1}:")
    train_loss,train_acc = train(train_dataloader,model,loss_fn,optimizer)
    val_loss, val_acc = val(test_dataloader,model,loss_fn)
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    if val_acc > min_acc:
        folder = "save_model"
        if not os.path.exists(folder):
            os.makedirs("save_model")
        min_acc = val_acc
        print(f"save best model,第{t+1}轮")
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t+1}#+start_epoch}
        torch.save(state,'save_model/best_model.pth')
    if t == EPOCH-1:
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': t + 1}# + start_epoch}
        torch.save(model.state_dict(), "save_model/last_model_128.pth")

matplot_loss(loss_train,loss_val)
matplot_acc(acc_train,acc_val)
print("Done!")

