import os
from random import random
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms,datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from Focal_Loss import FocalLoss
from torch.utils.data import DataLoader
import datetime
from random import seed
from net import MyNetwork
import time
import warnings

warnings.filterwarnings("ignore")

transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 使用torchvision.datasets.ImageFolder读取数据集 指定train 和 test文件夹
test_dataset = torchvision.datasets.ImageFolder('', transform=transform)  # test datasets path
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyNetwork().to(device)
model.load_state_dict(torch.load('C:/Users/DELL/Desktop/biaoqian/save_model/last_model_256.pth'))

classes = ['original','recaptured',]
show = ToPILImage()

model.eval()
start_time = time.time()
for i in range(1):
    x,y = test_dataset[i][0],test_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x,dim=0).float(),requires_grad = True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        predicted,actual = classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",Actual:"{actual}"')
end_time = time.time()
run_time = end_time - start_time
print(run_time)


