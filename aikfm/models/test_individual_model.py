import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from aikfm.dataset import AikfmDataset
from aikfm.models import CAN8
from torch.utils.data import DataLoader
from torchsummary import summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="Data root directory, with Training and Test directories.")
    parser.add_argument("--batch_size", default=20, type=int, help="Mini-batch size to train the model.")
    parser.add_argument("--results_dir", default=os.path.join(os.getenv('HOME'), 'ai-kfm-results'), type=str)

    args = parser.parse_args()

    num_of_gpus = torch.cuda.device_count()
    print(f'Available GPUs : {num_of_gpus}')

    mini_batch_size = args.batch_size
    lambda1 = args.lambda1
    lambda2 = args.lambda2

    dataset = AikfmDataset(args.data_root)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generater 1
    g1 = CAN8()
    g1.to(device)

    print(summary(g1, (1, 1200, 1600)))
    print(torch.cuda.memory_summary())
