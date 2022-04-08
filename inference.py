import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aikfm.dataset import AikfmDatasetInference
from aikfm.models import CAN8, UCAN64

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str, help="Data root directory, with Training and Test directories.")
    parser.add_argument("g1_ckpt_path", type=str, help="Generator 1, model checkpoint path.")
    parser.add_argument("g2_ckpt_path", type=str, help="Generator 2, model checkpoint path.")
    parser.add_argument("--batch_size", default=20, type=int, help="Mini-batch size to train the model.")
    parser.add_argument("--threshold", default=0.4, type=float, help="Thereshold for final mask creation, values < thresh will be 0, and rest will be 1")
    parser.add_argument("--results_dir", default=os.path.join(os.getenv('HOME'), 'ai-kfm-results'), type=str)
    parser.add_argument("--results_file", default='results-aikfm.csv', type=str)


    args = parser.parse_args()

    mini_batch_size = args.batch_size
    results_file = os.path.join(args.results_dir, args.results_file)
    g1_ckpt = torch.load(args.g1_ckpt_path) # Generator1 checkpoint
    g2_ckpt = torch.load(args.g2_ckpt_path) # Generator2 checkpoint

    dataset = AikfmDatasetInference(args.data_root)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generater 1
    g1 = nn.DataParallel(CAN8())
    g1.module.load_state_dict(g1_ckpt['model_state'])
    g1.to(device)

    # Generator 2
    g2 = nn.DataParallel(UCAN64())
    g2.module.load_state_dict(g2_ckpt['model_state'])
    g2.to(device)

    # Set all generators to eval
    g1.eval()
    g2.eval()

    total_it_per_epoch = len(dataloader)

    result = []
    for bt_idx, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device) # Move data to compute Device

        g1_out = g1(imgs)
        g1_out = torch.clamp(g1_out, 0.0, 1.0)
        g2_out = g2(imgs)
        g2_out = torch.clamp(g2_out, 0.0, 1.0)

        masks = (g1_out + g2_out)/2.0 # average of 2 generators
        masks = masks.squeeze(dim=1).permute(0, 2, 1).flatten(start_dim=1)
        masks = masks.cpu().detach().numpy() # move to cpu, and numpy

        _mask = masks<args.threshold
        masks[_mask] = 0
        masks[~_mask] = 1

        masks = masks.astype(np.int8)

        for i in range(len(targets)):
            mask = masks[i, :]
            label = ''
            flag = False
            l = 0
            for j, m in mask:
                if m == 1:
                    if not flag:
                        label += str(j+1)
                        l += 1
                        flag = True
                    else:
                        l += 1
                else:
                    if flag:
                        label += ' ' + str(l) + ' '
                        l = 0
                        flag = False
            label = label.strip()
            result += [targets[i], label]

    result_df = pd.DataFrame(result, columns=['ID', 'Mask'])
    result_df.to_csv(results_file, sep=',')

    print(f'Finished with the inference, and the results stored in {results_file}')
