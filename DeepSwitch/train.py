import os
import shutil
from tqdm import tqdm
import numpy as np
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
import torchvision.datasets as datasets

from eval import eval_net
from utils import get_lr, safe_div, AverageMeter, accuracy

import logging
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, "a")        
    handler.setFormatter(formatter)

    name = np.random.randint(2**32)
    logger = logging.getLogger(str(name))
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def run(model, net_size, root_dir, save_dir, input_size, batch_size, learning_rate, min_lr, epochs, device, patience, num_classes):
    model = model(input_size, num_classes)

    # Generate Necessary Files
    if os.path.isdir(save_dir):
        res = input(f"Directory already exists: {save_dir}\n1 - Use it anyway.\n2 - Delete.\n3 - Exit.\n>> ")
        if res == '2':
            shutil.rmtree(save_dir)
        if res == '3':
            exit(0)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Load Existing Models
    cks = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(save_dir)
           if re.match(r'(.*)\.(pth)', f)]
    cks.sort()
 
    if len(cks) > 0:
        epoch = cks[-1]
        latest = "model_save_epoch_{}.pth".format(epoch)

        print("Loading model Epoch:", epoch)
        model.load_state_dict(torch.load(os.path.join(save_dir, latest)))
    else:
        epoch = 0
    
    # Load Datasets
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(5),
        transforms.Resize(input_size[1:]),
        transforms.ToTensor(),
    ])
                                        
    #Load our dataset
    train_set = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=transform)
    val_set = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build the Network
    model = model.to(device)

    global_step = epoch * len(train_set)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=patience)
    criterion = torch.nn.CrossEntropyLoss()

    # Logging Settings
    logger = setup_logger(os.path.join(save_dir, "model_run.log"))
    writer = SummaryWriter(log_dir=save_dir, purge_step=global_step)

    info = f'''
    Starting training:
    Epochs:          {epochs}
    Batch Size:      {batch_size}
    Learning Rate:   {learning_rate}
    N. of Classes:   {num_classes}
    Input Size:      {input_size}
    Minimum LR:      {min_lr}
    Patience:        {patience}
    Training Size:   {len(train_set)}
    Validation Size: {len(val_set)}
    Save Directory:  {save_dir}
    Model Directory: {root_dir}
    Device:          {device.type}
    Network Size:    {net_size}
    '''
    logger.info(info)
    print(info)
    
    # Run the Model
    for epoch in range(epoch, epochs):
        model.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        with tqdm(total=len(train_set), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for (inputs, target) in train_loader:

                inputs = inputs.to(device=device)
                target = target.to(device=device)

                output = model(inputs)
                loss = criterion(output, target)

                acc1 = accuracy(output, target, topk=(1,))
                train_loss.update(loss.item(), inputs.size(0))
                train_acc.update(acc1[0], inputs.size(0))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Training Loss', loss.item(), global_step)
                writer.add_scalar('Training Accuracy', train_acc.val, global_step)
                
                pbar.set_postfix(**{'acc': train_acc.avg.item(), 'loss': train_loss.avg})
                pbar.update(inputs.shape[0])

                global_step += 1

            val_loss, val_acc = eval_net(model, val_loader, device, len(val_set), writer, global_step)
            scheduler.step(val_loss)

            writer.add_scalar('Validation Accuracy', val_acc, global_step)
            writer.add_scalar('Validation Loss', val_loss, global_step)
            writer.add_scalar('Learning Rate', get_lr(optimizer), global_step)

            if get_lr(optimizer) <= min_lr:
                logger.info('Minimum Learning Rate Reached: Early Stopping')
                break

            if (epoch % 5) == 0:
                writer.add_images('training', inputs[:1,:,:,:], global_step)
                torch.save(model.state_dict(), os.path.join(save_dir, "model_save_epoch_{}.pth".format(epoch)))
                logger.info('Checkpoint {} saved!'.format(epoch))
                logger.info('Validation Loss: {}'.format(val_loss))
                logger.info('Learning Rate: {}'.format(get_lr(optimizer)))

    writer.close()
    logger.info('Training finished, exiting...')
    torch.save(model.state_dict(), os.path.join(save_dir, "model_save_epoch_{}.pth".format(epoch)))
    logger.info('Final checkpoint {} saved!'.format(epoch))
    del model
