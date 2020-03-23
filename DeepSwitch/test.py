import os
import shutil
from tqdm import tqdm
import numpy as np
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

from eval import eval_net
from utils import get_lr, safe_div, AverageMeter
from model import DeepSwitch

def run(model, root_dir, save_dir, input_size, batch_size, device, num_classes, set_type):
    model = model(input_size, num_classes)

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
        transforms.Resize(input_size[1:]),
        transforms.ToTensor(),
    ])

    #Load our dataset
    test_set = datasets.ImageFolder(os.path.join(root_dir, set_type), transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    labels = list(test_set.class_to_idx.keys())
    values = list(test_set.class_to_idx.values())

    # Build the Network
    model = model.to(device)
    model.eval()

    # Logging Settings
    info = f'''
    Starting training:
    Batch Size:      {batch_size}
    N. of Classes:   {num_classes}
    Input Size:      {input_size}
    Save Directory:  {save_dir}
    Loaded Epoch:    {epoch}
    Test Size:       {len(test_set)}
    Model Directory: {root_dir}
    Device:          {device.type}
    '''
    print(info)
    
    # Run the Model
    acc = AverageMeter()
    matrix = np.zeros(shape=(len(values), len(values)))

    with tqdm(total=len(test_set), desc='Testing', unit='img', leave=False) as pbar:
        for batch in test_loader:
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
    save_dir = 'runs/v31'
    set_type = 'valid'
    input_size = (3, 384//2, 512//2)
    num_classes = 7
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run(DeepSwitch, root_dir, save_dir, input_size, batch_size, device, num_classes, set_type)
