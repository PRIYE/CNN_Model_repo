import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom_Res_Net(nn.Module):
    def __init__(self):
        super(Custom_Res_Net, self).__init__()
        # Prep layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),  #32>>32 | 1>>3 | 1>>1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Layer 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False), #32>>32 | 3>>5 | 1>>1
            nn.MaxPool2d(2,2), #32>>17 | 5>>6 | 1>>2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.res1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False), #17>>17 | 6>>10 | 2>>2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False), #17>>17 | 10>>14 | 2>>2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

        )# Layer 1 output -  6, 14
        # Layer 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False), #17>>17 | 14>>18 | 2>>2
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2,2), #17>>9.5 | 18>>20 | 2>>4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),

        )# Layer 2 output -  12, 20
        # Layer 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False), #10>>10 | 20>>28 | 4>>4
            nn.MaxPool2d(2,2), #10>>6 | 28>>32 | 4>>8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

        )
        self.res2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False), #6>>6 | 32>>48 | 8>>8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False), #6>>6 | 48>>64 | 8>>8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

        )# Layer 3 output -  24, 32, 56 ,64
        # Layer 4
        self.conv5 = nn.Sequential(
            nn.MaxPool2d(4,4), #6>>3 | 64>>88 | 4>>8

        )# Layer 4 output -  48, 56, 80 ,88

        self.fc = nn.Sequential(
            nn.Conv2d(512, 10, 1, stride=1, padding=0, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(1),
            #nn.Dropout(0.05),
            #nn.Linear(10, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + self.res2(x)
        x = self.conv5(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)