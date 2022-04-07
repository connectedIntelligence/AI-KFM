import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from aikfm.dataset import AikfmDataset
from aikfm.models import CAN8, UCAN64, discriminator

max_epoch_num = 30
mini_batch_size = 20
lambda1 = 100
lambda2 = 10

if __name__ == "__main__":

    dataset = AikfmDataset("~/DKLabs/AI-KFM/AI-KFM/data")
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generater 1
    g1 = CAN8()
    g1.to(device)

    # Generator 2
    g2 = UCAN64()
    g2.to(device)

    # Discriminator
    dis = discriminator()
    dis.to(device)

    # Define optimizers
    optim_g1 = optim.AdamW(g1.parameters(), lr=1e-4, weight_decay=1e-5)
    optim_g2 = optim.AdamW(g2.parameters(), lr=1e-4, weight_decay=1e-5)
    optim_dis = optim.AdamW(dis.parameters(), lr=1e-5, weight_decay=1e-6)

    # Loss function
    loss1 = nn.BCEWithLogitsLoss()

    it = 0
    for epoch in range(max_epoch_num):
        # LR decay
        if (epoch+1)%10 == 0:
            for p in optim_g1.param_groups:
                p['lr'] = 0.2
            for q in optim_g2.param_groups:
                q['lr'] = 0.2
            for r in optim_dis.param_groups:
                r['lr'] = 0.2

        print(f'Training on epoch {epoch+1}')
        total_it_per_epoch = len(dataloader)

        for bt_idx, (imgs, masks) in enumerate(dataloader):
            imgs, masks = imgs.to(device), masks.to(device) # Move data to compute Device

            it += 1
            print(f'current iteration {bt_idx+1}/{total_it_per_epoch}, epoch {epoch+1}/{max_epoch_num}, total iteration: {it}, g1 lr: {float(0)}, g2 lr: {float(0)}, Dis lr: {float(0)}')

            ###############################
            # Train the discriminator first
            dis.train()
            g1.eval()
            g2.eval()
            optim_g1.zero_grad()
            optim_g2.zero_grad()
            optim_dis.zero_grad()

            # Get generator outputs
            g1_out = g1(imgs) # [B, 1, 1200, 1600]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(imgs) # [B, 1, 1200, 1600]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([imgs, 2 * masks - 1], dim = 1) # [B, 4, H, W]
            neg1 = torch.cat([imgs, 2 * g1_out - 1], dim = 1) # [B, 4, H, W]
            neg2 = torch.cat([imgs, 2 * g2_out - 1], dim = 1) # [B, 4, H, W]

            dis_input = torch.cat([pos1, neg1, neg2], dim=0) # # [3*B, 4, H, W]

            # Get discriminator output
            logits_real, logits_fake1, logits_fake2, Lgc = dis(dis_input)

            const1 = torch.ones(imgs.size(0), 1, device=device, dtype=torch.float32)
            const0 = torch.zeros(imgs.size(0), 1, device=device, dtype=torch.float32)

            gen_gt = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            ES0 = torch.mean(loss1(logits_real, gen_gt))
            ES1 = torch.mean(loss1(logits_fake1, gen_gt1))
            ES2 = torch.mean(loss1(logits_fake2, gen_gt2))

            dis_loss = ES0 + ES1 + ES2 # Discriminator loss
            print(f'Discriminator loss : {dis_loss}')

            dis_loss.backward() # Compute gradients
            optim_dis.step() # Apply gradients


            #########################
            # Train generator g1
            dis.eval()
            g1.train()
            g2.eval()
            optim_g1.zero_grad()
            optim_g2.zero_grad()
            optim_dis.zero_grad()

            g1_out = g1(imgs) # [B, 1, H, W]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)
            MD1 = torch.mean(torch.mul(torch.pow(g1_out - masks, 2), masks))
            FA1 = torch.mean(torch.mul(torch.pow(g1_out - masks, 2), 1 - masks))
            MF_loss1 = lambda1 * MD1 + FA1

            g2_out = g2(imgs) # [B, 1, H, W]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([imgs, 2 * masks - 1], dim = 1) # [B, 4, H, W]
            neg1 = torch.cat([imgs, 2 * g1_out - 1], dim = 1) # [B, 4, H, W]
            neg2 = torch.cat([imgs, 2 * g2_out - 1], dim = 1) # [B, 4, H, W]

            dis_input = torch.cat([pos1, neg1, neg2], dim=0) # # [3*B, 4, H, W]

            # Get discriminator output
            logits_real, logits_fake1, logits_fake2, Lgc = dis(dis_input) # [B, 3] [B, 3] [B, 3]

            const1 = torch.ones(imgs.size(0), 1, device=device, dtype=torch.float32)
            const0 = torch.zeros(imgs.size(0), 1, device=device, dtype=torch.float32)

            gen_gt = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            gen_adv_loss1 = torch.mean(loss1(logits_fake1, gen_gt))
            gen_loss1 = 100*MF_loss1 + 10*gen_adv_loss1 + 1*Lgc
            print(f'Generator1 loss : {gen_loss1}')

            gen_loss1.backward()
            optim_g1.step()


            #########################
            # Train generator g2
            dis.eval()
            g1.eval()
            g2.train()
            optim_g1.zero_grad()
            optim_g2.zero_grad()
            optim_dis.zero_grad()

            g1_out = g1(imgs) # [B, 1, H, W]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(imgs) # [B, 1, H, W]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            MD2 = torch.mean(torch.mul(torch.pow(g2_out - masks, 2), masks))
            FA2 = torch.mean(torch.mul(torch.pow(g2_out - masks, 2), 1 - masks))
            MF_loss2 = MD2 + lambda2 * FA1

            pos1 = torch.cat([imgs, 2 * masks - 1], dim = 1) # [B, 4, H, W]
            neg1 = torch.cat([imgs, 2 * g1_out - 1], dim = 1) # [B, 4, H, W]
            neg2 = torch.cat([imgs, 2 * g2_out - 1], dim = 1) # [B, 4, H, W]

            dis_input = torch.cat([pos1, neg1, neg2], dim=0) # # [3*B, 4, H, W]

            # Get discriminator output
            logits_real, logits_fake1, logits_fake2, Lgc = dis(dis_input) # [B, 3] [B, 3] [B, 3]

            const1 = torch.ones(imgs.size(0), 1, device=device, dtype=torch.float32)
            const0 = torch.zeros(imgs.size(0), 1, device=device, dtype=torch.float32)

            gen_gt = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            gen_adv_loss2 = torch.mean(loss1(logits_fake2, gen_gt))
            gen_loss2 = 100*MF_loss2 + 10*gen_adv_loss2 + 1*Lgc
            print(f'Generator2 loss : {gen_loss2}')

            gen_loss2.backward()
            optim_g2.step()