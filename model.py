import torch
import torch.nn as nn


class MyAlexNet(nn.Module):

    def __init__(self, classes=100):
        super(MyAlexNet, self).__init__()
        # cifar100 100类，每张图片大小3*32*32
        self.features_extract = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1),  # 3*32*32-->96*30*30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -->96*14*14

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # -->256*9*9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -->256*4*4

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # -->256*3*3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # -->384*3*3
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # -->256*3*3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # -->256*2*2=1024
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, classes)
        )

    def forward(self, x):
        x = self.features_extract(x)
        x = torch.flatten(x, 1)
        # x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.classifier(x)
        return x
