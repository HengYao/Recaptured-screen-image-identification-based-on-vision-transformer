import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.deepvit import DeepViT
import torchvision
import warnings

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(5, 9, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(9, 16, kernel_size=5, stride=2, padding=1, bias=False)
        self.conv5 = nn.Conv2d(16,32, kernel_size=5, stride=2, padding=1, bias=False)
        self.advpool2d = nn.AdaptiveAvgPool2d(256)
        self.pool = nn.AdaptiveAvgPool2d(32)
        self.bn1 = nn.BatchNorm2d(9)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.PReLU()
        self.v = v = DeepViT(
            image_size=256,
            patch_size=16,
            num_classes=1000,
            dim=1024,
            depth=1,
            heads=32,
            mlp_dim=512,
            dropout=0.9,
            emb_dropout=0.9
        )

        self.flatten = nn.Flatten()
        self.LC1 = nn.Linear(1000, 2048)
        self.LC5 = nn.Linear(2048,1024)
        self.LC4 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.advpool2d(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.v(x)
        x = self.flatten(x)
        x = self.LC1(x)
        x = F.dropout(x,p=0.9)
        x = self.relu(x)
        x = self.LC5(x)
        x = F.dropout(x,p=0.9)
        x = self.relu(x)
        x = self.LC4(x)
        x = torch.softmax(x, 1)
        return x

