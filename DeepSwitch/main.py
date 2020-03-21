import torch

import train
from model import DeepSwitch


# Declare Global Settings
root_dir = "/media/luigifcruz/HDD1/SETI/fft_signal"
input_size = (3, 384//2, 512//2)
num_classes = 7
learning_rate = 1e-5
min_lr = 1e-6
patience = 5
batch_size = 8
epochs = 750
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Declare Interation Settings
interations = [
    {"model": DeepSwitch, "save_dir": 'runs/v13', "net_size": 1},
]

if __name__ == "__main__":
    for data in interations:
        train.run(data['model'], data['net_size'], root_dir, data['save_dir'],
                  input_size, batch_size, learning_rate, min_lr, epochs, device, patience, num_classes)
