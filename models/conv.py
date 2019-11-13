import torch.nn as nn
import torch
#import torch.utils.model_zoo as model_zoo
import math

__all__ = ['conv']

class conv(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l1 = nn.Linear(256, 512)
        self.relu4 = nn.ReLU(True)
        self.l2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.max1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.max2(x)
        

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.max3(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.l1(x)
        x = self.relu4(x)
        x = self.l2(x)
        return x

if __name__ == "__main__":
    a = torch.zeros((1,1,32,32))
    model = conv(num_classes = 10, in_channels = 1)
    res = model(a)
    print(res.shape)

