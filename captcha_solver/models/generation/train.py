import torch
import torch.nn as nn
from torch import autograd
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

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    d_interpolated = D(interpolated)
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    gradients = autograd.grad(
        outputs=d_interpolated, inputs=interpolated, grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty

data_dir = "colorcaptcha"
batch_size = 64
latent_dim = 100
num_epochs = 100
lr_G = 0.00002
lr_D = 0.00001
lambda_gp = 10
lambda_l1 = 100
pretrain_epochs = 500 

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

fake_buffer = ImageBuffer(max_size=50)

for epoch in range(num_epochs):
    dataset = CaptchaDataset(data_dir, sample_size=4000)
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
        loss_D = fake_pred.mean() - real_pred.mean()
        gp = compute_gradient_penalty(D, real_images, fake_images_for_D, device)
        d_loss = loss_D + lambda_gp * gp

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        for p in D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True

        for _ in range(3):
            z = torch.randn(cur_batch, latent_dim, device=device)
            if epoch < pretrain_epochs:
                text_for_G = text_onehot      
            else:
                text_for_G = text_input_G    
            gen_images = G(z, text_for_G)
            gen_pred = D(gen_images)
            loss_G_adv = -gen_pred.mean()
            loss_G_L1 = 0
            if epoch < pretrain_epochs:
                loss_G_L1 = criterion_L1(gen_images, real_images)
            g_loss = loss_G_adv + (lambda_l1 * loss_G_L1 if epoch < pretrain_epochs else 0)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

    G.eval()
    if not os.path.exists('result'):
        os.makedirs('result')
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
            save_image(gen_img, f"result/epoch{epoch+1}_sample{j}_{rand_text}.png", normalize=True)
    G.train()

    print(f"Epoch [{epoch+1}/{num_epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {loss_G_adv.item():.4f}")
