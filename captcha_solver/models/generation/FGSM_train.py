import torch
import torch.nn as nn
from torch import autograd
from torch.utils.data import DataLoader
import random

from FGSM_dataset import CaptchaDataset
from FGSM_generator import Generator
from FGSM_discriminator import Discriminator
from solver import Solver

data_dir = "colorcaptcha"
batch_size = 64
latent_dim = 100
num_epochs = 100
lr_G = 0.0002
lr_D = 0.0001
lr_solver = 0.001    
lambda_gp = 10      
lambda_l1 = 100      
pretrain_epochs = 5    
lambda_solver = 1.0    

chars = [str(d) for d in range(10)] + [chr(c) for c in range(ord('A'), ord('Z')+1)] + [chr(c) for c in range(ord('a'), ord('z')+1)]
num_chars = len(chars)     
blank_idx = num_chars         
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

dataset = CaptchaDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(latent_dim=latent_dim, label_length=5, num_chars=num_chars).to(device)
D = Discriminator().to(device)
solver = Solver(num_chars=num_chars, blank_idx=blank_idx).to(device)

optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
optimizer_solver = torch.optim.Adam(solver.parameters(), lr=lr_solver, betas=(0.5, 0.999))

criterion_L1 = nn.L1Loss()
ctc_loss_fn = nn.CTCLoss(blank=blank_idx, reduction='mean')  

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

def compute_gradient_penalty(D_net, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    d_interpolated = D_net(interpolated)
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    gradients = autograd.grad(outputs=d_interpolated, inputs=interpolated,
                              grad_outputs=grad_outputs, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty

def decode_solver_output(log_probs, blank_idx):
    decoded_results = []
    pred_indices = log_probs.argmax(dim=2)  
    pred_indices = pred_indices.cpu().numpy()
    T, N = pred_indices.shape
    for n in range(N):
        seq = []
        prev_char_idx = None
        for t in range(T):
            idx = int(pred_indices[t, n])
            if idx == blank_idx:
                prev_char_idx = None
                continue
            if idx == prev_char_idx:
                continue  
            seq.append(idx)
            prev_char_idx = idx
        text = "".join(idx_to_char[i] for i in seq)
        decoded_results.append(text)
    return decoded_results

fake_buffer = ImageBuffer(max_size=50)
epsilon = 0.05       
epsilon_min = 0.0
epsilon_max = 0.2
use_fgsm_d = False   
for epoch in range(num_epochs):
    solver.train()  
    for i, (real_images, labels) in enumerate(dataloader):
        real_images = real_images.to(device)
        cur_batch = real_images.size(0)
        D.train(); G.train()
        for p in D.parameters(): 
            p.requires_grad = True
        for p in G.parameters(): 
            p.requires_grad = False

        label_onehot = torch.zeros(cur_batch, 5, num_chars, device=device)
        for idx, text in enumerate(labels):
            for j, ch in enumerate(text):
                if ch in char_to_idx:
                    label_idx = char_to_idx[ch]
                    label_onehot[idx, j, label_idx] = 1
        label_onehot = label_onehot.view(cur_batch, -1)

        z = torch.randn(cur_batch, latent_dim, device=device)
        fake_images = G(z, label_onehot).detach()
        fake_images_for_D = fake_buffer.push_and_get(fake_images) 

        real_pred = D(real_images)
        fake_pred = D(fake_images_for_D)
        loss_D = fake_pred.mean() - real_pred.mean()
        gp = compute_gradient_penalty(D, real_images, fake_images_for_D)
        d_loss = loss_D + lambda_gp * gp

        if use_fgsm_d:
            real_images.requires_grad_(True)
            real_score = D(real_images).mean()         
            grad_real = autograd.grad(-real_score, real_images, create_graph=False)[0]  
            perturb_real = epsilon * grad_real.sign()
            real_images_adv = (real_images + perturb_real).clamp(-1, 1).detach()
            fake_images_for_D.requires_grad_(True)
            fake_score = D(fake_images_for_D).mean()
            grad_fake = autograd.grad(fake_score, fake_images_for_D, create_graph=False)[0]
            perturb_fake = epsilon * grad_fake.sign()
            fake_images_adv = (fake_images_for_D + perturb_fake).clamp(-1, 1).detach()
            real_pred_adv = D(real_images_adv)
            fake_pred_adv = D(fake_images_adv)
            loss_D_adv = fake_pred_adv.mean() - real_pred_adv.mean()
            d_loss = 0.5 * (loss_D + loss_D_adv) + lambda_gp * gp

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        solver.train()
        for p in solver.parameters():
            p.requires_grad = True
        for p in G.parameters():
            p.requires_grad = False
        for p in D.parameters():
            p.requires_grad = False

        target_indices = []
        target_lengths = []
        for text in labels:
            target_lengths.append(len(text))
            for ch in text:
                if ch in char_to_idx:
                    target_indices.append(char_to_idx[ch])
        targets = torch.tensor(target_indices, dtype=torch.long).to(device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(device)
        input_lengths = torch.full(size=(cur_batch,), fill_value=32, dtype=torch.long).to(device) 

        real_images_solver = real_images.detach().clone().requires_grad_(True)
        log_probs = solver(real_images_solver)
        solver_loss_real = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        grad = autograd.grad(solver_loss_real, real_images_solver, retain_graph=True)[0]
        perturb = epsilon * grad.sign()
        adv_images = (real_images_solver + perturb).clamp(-1, 1)
        log_probs_adv = solver(adv_images)
        solver_loss_adv = ctc_loss_fn(log_probs_adv, targets, input_lengths, target_lengths)
        solver_loss_total = 0.5 * (solver_loss_real + solver_loss_adv)
        optimizer_solver.zero_grad()
        solver_loss_total.backward()
        optimizer_solver.step()


        z_solver = torch.randn(cur_batch, latent_dim, device=device)
        fake_images_solver = G(z_solver, label_onehot).detach()
        log_probs_fake = solver(fake_images_solver)
        solver_loss_fake = ctc_loss_fn(log_probs_fake, targets, input_lengths, target_lengths)
        optimizer_solver.zero_grad()
        solver_loss_fake.backward()
        optimizer_solver.step()


        for p in solver.parameters():
            p.requires_grad = False
        for p in D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True

        for _ in range(3):
            z = torch.randn(cur_batch, latent_dim, device=device)
            gen_images = G(z, label_onehot)
            gen_pred = D(gen_images)
            loss_G_adv = -gen_pred.mean()
            log_probs_gen = solver(gen_images)  
            solver_loss_gen = ctc_loss_fn(log_probs_gen, targets, input_lengths, target_lengths)
            loss_G_solver = - solver_loss_gen 
            if epoch < pretrain_epochs:
                loss_L1 = criterion_L1(gen_images, real_images)
                g_loss = loss_L1 * lambda_l1
            else:
                g_loss = loss_G_adv + lambda_solver * loss_G_solver

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
    solver.eval(); G.eval()
    total, correct = 0, 0
    eval_batches = 2 
    for _ in range(eval_batches):
        eval_batch = batch_size
        z_eval = torch.randn(eval_batch, latent_dim, device=device)
        eval_labels = []
        for _ in range(eval_batch):
            rand_text = "".join(random.choice(chars) for _ in range(5))
            eval_labels.append(rand_text)
        eval_onehot = torch.zeros(eval_batch, 5, num_chars, device=device)
        for idx, text in enumerate(eval_labels):
            for j, ch in enumerate(text):
                if ch in char_to_idx:
                    eval_onehot[idx, j, char_to_idx[ch]] = 1
        eval_onehot = eval_onehot.view(eval_batch, -1)
        gen_imgs = G(z_eval, eval_onehot)
        log_probs_eval = solver(gen_imgs)
        decoded_texts = decode_solver_output(log_probs_eval, blank_idx)
        for pred_text, true_text in zip(decoded_texts, eval_labels):
            if pred_text == true_text:
                correct += 1
            total += 1
    solver_acc = correct / total if total > 0 else 0.0
    if solver_acc > 0.7:
        epsilon = min(epsilon + 0.005, epsilon_max)
    elif solver_acc < 0.3:
        epsilon = max(epsilon - 0.005, epsilon_min)

    print(f"Epoch [{epoch+1}/{num_epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {loss_G_adv.item():.4f}  "
          f"Solver_acc: {solver_acc*100:.2f}%  epsilon: {epsilon:.4f}")
