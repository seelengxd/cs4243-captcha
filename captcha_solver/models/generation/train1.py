import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import random

from dataset import CaptchaDataset
from generator import Generator
from discriminator import Discriminator

class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []
    def push_and_get(self, images):
        out_images = []
        for img in images:
            img = img.detach()
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                out_images.append(img)
            else:
                if torch.rand(1).item() < 0.5:
                    idx = torch.randint(0, len(self.buffer), (1,)).item()
                    old_img = self.buffer[idx]
                    self.buffer[idx] = img
                    out_images.append(old_img)
                else:
                    out_images.append(img)
        return torch.stack(out_images)

data_dir = "colorcaptcha"
batch_size = 64
latent_dim = 100
num_epochs = 100
lr_G = 0.00002
lr_D = 0.00001
lambda_l1 = 100
pretrain_epochs = 50  

temp_dataset = CaptchaDataset(data_dir, sample_size=0)
char_set = temp_dataset.char_set
print(char_set)
char_to_idx = temp_dataset.char_to_idx
max_length = temp_dataset.max_length
text_dim = len(char_set) * max_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(latent_dim=latent_dim, text_dim=text_dim).to(device)
D = Discriminator().to(device)
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
criterion_L1 = nn.L1Loss()
criterion_adv = nn.BCEWithLogitsLoss()

fake_buffer = ImageBuffer(max_size=50)

for epoch in range(num_epochs):
    dataset = CaptchaDataset(data_dir, sample_size=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for i, (real_images, text_onehot) in enumerate(dataloader):
        real_images = real_images.to(device)
        text_onehot = text_onehot.to(device)
        cur_batch = real_images.size(0)
        D.train()
        G.train()
        for p in D.parameters():
            p.requires_grad = True
        for p in G.parameters():
            p.requires_grad = False

        if epoch < pretrain_epochs:
            text_input_G = text_onehot             
        else:
            random_texts = [''.join(random.choices(char_set, k=max_length)) for _ in range(cur_batch)]
            random_onehots = []
            for txt in random_texts:
                oh = torch.zeros((max_length, len(char_set)), dtype=torch.float)
                for j, ch in enumerate(txt):
                    if ch in char_to_idx:
                        oh[j, char_to_idx[ch]] = 1.0
                random_onehots.append(oh.view(-1))
            text_input_G = torch.stack(random_onehots).to(device)

        z = torch.randn(cur_batch, latent_dim, device=device)
        fake_images = G(z, text_input_G).detach()             
        fake_images_for_D = fake_buffer.push_and_get(fake_images)  

        real_pred = D(real_images)
        fake_pred = D(fake_images_for_D)
        real_loss = criterion_adv(real_pred, torch.ones_like(real_pred))
        fake_loss = criterion_adv(fake_pred, torch.zeros_like(fake_pred))
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        for p in D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True

        for _ in range(3): 
            z = torch.randn(cur_batch, latent_dim, device=device)
            text_for_G = text_onehot if epoch < pretrain_epochs else text_input_G
            gen_images = G(z, text_for_G)
            gen_pred = D(gen_images)
            loss_G_adv = criterion_adv(gen_pred, torch.ones_like(gen_pred))
            loss_G_L1 = 0
            if epoch < pretrain_epochs:
                loss_G_L1 = criterion_L1(gen_images, real_images)
            g_loss = loss_G_adv + (lambda_l1 * loss_G_L1 if epoch < pretrain_epochs else 0)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

    G.eval()
    if not os.path.exists('C_result'):
        os.makedirs('C_result')
    with torch.no_grad():
        for j in range(10):
            rand_text = ''.join(random.choices(char_set, k=max_length))
            rand_onehot = torch.zeros((max_length, len(char_set)), dtype=torch.float)
            for k, ch in enumerate(rand_text):
                if ch in char_to_idx:
                    rand_onehot[k, char_to_idx[ch]] = 1.0
            rand_onehot = rand_onehot.view(1, -1).to(device)
            z = torch.randn(1, latent_dim, device=device)
            gen_img = G(z, rand_onehot)
            save_image(gen_img, f"C_result/epoch{epoch+1}_sample{j}_{rand_text}.png", normalize=True)
    G.train()

    print(f"Epoch [{epoch+1}/{num_epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {loss_G_adv.item():.4f}")
