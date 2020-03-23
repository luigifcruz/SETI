import torch.nn as nn
import torch


class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.conv(x)


class dual_conv_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(dual_conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.conv(x)


class DeepSwitch(nn.Module):

    def __init__(self, input_size, num_classes):
        super(DeepSwitch, self).__init__()

        self.Conv1 = dual_conv_block(input_size[0], 8)
        self.Conv2 = dual_conv_block(8, 16)
        self.Conv3 = dual_conv_block(16, 32)
        self.Conv4 = conv_block(32, 64)
        self.Conv5 = conv_block(64, 128)
        self.Conv6 = conv_block(128, 256)

        size = (input_size[1]//64) * (input_size[2]//64) * 256

        self.fc = nn.Sequential(
            nn.Linear(size, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes), 
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x