import torch.nn as nn
import torch

def make_features(cfg, batch_norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class DeepSwitch(nn.Module):

    def __init__(self, cfg, num_classes, batch_norm=False):
        super(DeepSwitch, self).__init__()

        self.features = make_features(cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 4))
        self.classifier = nn.Sequential(
            nn.Linear(cfg[-2] * 3 * 4, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes), 
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
