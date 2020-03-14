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
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.conv(x)


class DeepSwitch(nn.Module):

    def __init__(self, input_size, num_classes):
        super(DeepSwitch, self).__init__()

        self.Conv1 = conv_block(input_size[0], 32)
        self.Conv2 = conv_block(32, 64)

        size = (input_size[1]//4) * (input_size[2]//4) * 64

        self.fc = nn.Sequential(
            nn.Linear(size, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes), 
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x