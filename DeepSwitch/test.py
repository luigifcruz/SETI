import os
import shutil
from tqdm import tqdm
import numpy as np
import time
import re
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
import torchvision.datasets as datasets
import torchvision.models as models

from eval import eval_net
from utils import get_lr, safe_div, AverageMeter
from model import DeepSwitch

def run(cfg, bn, root_dir, save_dir, input_size, batch_size, device, num_classes, set_type):
    # VGG-16
    #model = models.vgg16(pretrained=False)
    #model.classifier[6] = nn.Linear(4096, num_classes)

    # Inception V3
    #model = models.inception_v3(pretrained=False, aux_logits=False)
    #model.fc = nn.Linear(2048, num_classes)

    # SqueezeNet
    #model = models.squeezenet1_0(pretrained=False)
    #model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

    # Resenet18
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, num_classes)

    #model = DeepSwitch(cfg, num_classes, batch_norm=bn)

    # Load Existing Models
    cks = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(save_dir)
           if re.match(r'(.*)\.(pth)', f)]
    cks.sort()
 
    if len(cks) > 0:
        epoch = cks[-1]
        latest = "model_save_epoch_{}.pth".format(epoch)

        model.load_state_dict(torch.load(os.path.join(save_dir, latest)))
    else:
        print("Couldn't find a save file.")
        exit()

    # Load Datasets
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    #Load our dataset
    test_set = datasets.ImageFolder(os.path.join(root_dir, set_type), transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    labels = list(test_set.class_to_idx.keys())
    values = list(test_set.class_to_idx.values())

    # Build the Network
    model = model.to(device)
    model.eval()

    # Logging Settings
    print(f'''
    Starting training:
    Batch Size:      {batch_size}
    N. of Classes:   {num_classes}
    Input Size:      {input_size}
    Save Directory:  {save_dir}
    Loaded Epoch:    {epoch}
    Test Size:       {len(test_set)}
    Model Directory: {root_dir}
    Device:          {device.type}
    ''')
    
    # Run the Model
    acc = AverageMeter()
    speed = AverageMeter()
    matrix = np.zeros(shape=(len(values), len(values)))

    
    with tqdm(total=len(test_set), desc='Testing', unit='img', leave=False) as pbar:
        for batch in test_loader:
            st = time.time()
            inputs, target = batch
            
            inputs = inputs.to(device=device)
            target = target.to(device=device)

            output = model(inputs)
            _, pred = torch.max(output, dim=1)

            pred = pred.detach().cpu()
            target = target.detach().cpu()

            # Accumulate Confusion Matrix
            matrix += confusion_matrix(pred, target, labels=values)

            # Calculate Accuracy
            a = accuracy_score(pred, target)
            acc.update(a, inputs.size(0))

            pbar.set_postfix(**{'acc': acc.avg.item()})
            pbar.update(inputs.shape[0])
            et = time.time()

            speed.update(batch_size/(et-st))

    print(speed.avg)

    df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(15, 12))
    sn.heatmap(df_cm, annot=True, cmap="OrRd")
    plt.title('Confusion Matrix\nDataset: {} | Accuracy: {:.2f}'.format(set_type, acc.avg.item()*100))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("test_cm.png")

    print(acc.avg.item())
    print(matrix)

if __name__ == "__main__":
    root_dir = '/media/luigifcruz/HDD1/SETI/fft_signal'
    set_type = 'valid'
    num_classes = 7
    batch_size = 64
    bn = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    interations = [
        {
            'save_dir': 'runs/v38',
            'size': (192, 256), 
            'cfg': [8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M'],
            'bn': True,
        }
    ]

    for data in interations:
        run(data['cfg'], data['bn'], root_dir, data['save_dir'], data['size'], batch_size, device, num_classes, set_type)
