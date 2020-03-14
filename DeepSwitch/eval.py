from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import utils

from utils import AverageMeter, accuracy

def eval_net(net, loader, device, n_val, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    
    loss = AverageMeter()
    acc = AverageMeter()

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            inputs, target = batch

            inputs = inputs.to(device=device)
            target = target.to(device=device)

            output = net(inputs)
            
            l = F.cross_entropy(output, target)
            loss.update(l.item(), inputs.size(0))

            a = accuracy(output, target, topk=(1,))
            acc.update(a[0], inputs.size(0))

            pbar.set_postfix(**{'acc': acc.avg.item()})
            pbar.update(inputs.shape[0])

    return loss.avg, acc.avg