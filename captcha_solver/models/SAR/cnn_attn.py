import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import math

class CaptchaDataset(Dataset):
    def __init__(self, img_dir, char_to_idx=None, idx_to_char=None, transform=None):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        self.img_paths.sort()
        if char_to_idx is None:
            chars = set()
            for path in self.img_paths:
                label = os.path.splitext(os.path.basename(path))[0]
                chars.update(list(label))
            chars = sorted(list(chars))
            self.char_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
            for i, ch in enumerate(chars):
                self.char_to_idx[ch] = i + 3
            self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
            self.idx_to_char[0] = '<PAD>'
            self.idx_to_char[1] = '<SOS>'
            self.idx_to_char[2] = '<EOS>'
        else:
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
        self.vocab_size = len(self.char_to_idx)
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(), transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        label_str = os.path.splitext(os.path.basename(img_path))[0].split('-')[0]
        return img_tensor, label_str

data_dir = "cs4243-captcha-main\\cs4243-captcha-main\\captcha_solver\\data\\train_cleaned_color_resized"
test_dir = "cs4243-captcha-main\\cs4243-captcha-main\\captcha_solver\\data\\test_cleaned_color_resized"
test_dataset = CaptchaDataset(test_dir)
dataset = CaptchaDataset(data_dir)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_dataset.char_to_idx = dataset.char_to_idx
train_dataset.idx_to_char = dataset.idx_to_char
val_dataset.char_to_idx = dataset.char_to_idx
val_dataset.idx_to_char = dataset.idx_to_char

def captcha_collate_fn(batch):
    images, dec_inputs, dec_targets = [], [], []
    max_len = 0
    pad_idx = dataset.char_to_idx['<PAD>']
    sos_idx = dataset.char_to_idx['<SOS>']
    eos_idx = dataset.char_to_idx['<EOS>']
    for img, label_str in batch:
        char_indices = [dataset.char_to_idx[ch] for ch in label_str]
        dec_in = [sos_idx] + char_indices           
        dec_out = char_indices + [eos_idx]          
        max_len = max(max_len, len(dec_out))
        images.append(img)
        dec_inputs.append(dec_in)
        dec_targets.append(dec_out)
    for i in range(len(dec_inputs)):
        dec_inputs[i] += [eos_idx] * (max_len - len(dec_inputs[i]))
        dec_targets[i] += [pad_idx] * (max_len - len(dec_targets[i]))
    images_tensor = torch.stack(images)
    dec_input_tensor = torch.tensor(dec_inputs, dtype=torch.long)
    dec_target_tensor = torch.tensor(dec_targets, dtype=torch.long)
    return images_tensor, dec_input_tensor, dec_target_tensor

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=captcha_collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=captcha_collate_fn)

print(f"总样本数: {len(dataset)}, 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
fig, axes = plt.subplots(1, 5, figsize=(15,3))
for i in range(5):
    img_tensor, label_str = dataset[i]
    img_np = img_tensor.numpy().squeeze()
    axes[i].imshow(img_np, cmap='gray')
    axes[i].set_title(f"Label: {label_str}")
    axes[i].axis('off')
plt.tight_layout()
plt.show()

class CRNN_NoLSTM(nn.Module):
    def __init__(self, num_classes):
        super(CRNN_NoLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d((2,1)),   
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
    def forward(self, x):
        x = self.cnn(x)                    
        x = self.adaptive_pool(x)           
        x = x.squeeze(2)                     
        x = x.permute(2, 0, 1)              
        return x

num_classes = dataset.vocab_size  
model = CRNN_NoLSTM(num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CTCLoss(blank=dataset.char_to_idx['<PAD>'], zero_infinity=True)

import math
num_epochs = 50
warmup_epochs = 5
base_lr = 1e-3
min_lr = 1e-5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        t = epoch - warmup_epochs
        T = num_epochs - warmup_epochs
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t / T))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for images, dec_input, dec_target in train_loader:
        images = images.to(device)
        target_list = []
        target_lengths = []
        for seq in dec_target:
            seq = seq.tolist()
            if dataset.char_to_idx['<EOS>'] in seq:
                eos_pos = seq.index(dataset.char_to_idx['<EOS>'])
                seq = seq[:eos_pos]
            target_list.extend(seq)
            target_lengths.append(len(seq))
        target_tensor = torch.tensor(target_list, dtype=torch.long)
        logits = model(images)
        T_out, batch_size, _ = logits.size()
        input_lengths = torch.full(size=(batch_size,), fill_value=T_out, dtype=torch.long)
        loss = criterion(logits.log_softmax(2), target_tensor, input_lengths, torch.tensor(target_lengths))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, LR: {lr:.6f}, Train Loss: {avg_loss:.4f}")

    model.eval()
    seq_correct = 0
    seq_total = 0
    char_correct = 0
    char_total = 0
    with torch.no_grad():
        for images, dec_input, dec_target in train_loader:
            images = images.to(device)
            logits = model(images)  
            log_probs = logits.log_softmax(2)
            _, pred_indices = torch.max(log_probs, dim=2)  
            pred_indices = pred_indices.permute(1, 0)        
            for i in range(pred_indices.size(0)):
                pred_seq = pred_indices[i].tolist()
                pred_text = ""
                prev = None
                for idx in pred_seq:
                    if idx != dataset.char_to_idx['<PAD>'] and idx != prev:
                        if idx == dataset.char_to_idx['<EOS>']:
                            break
                        pred_text += dataset.idx_to_char[idx]
                    prev = idx
                pred_text = pred_text.lower()  
                true_seq = dec_target[i].tolist()
                if dataset.char_to_idx['<EOS>'] in true_seq:
                    eos_pos = true_seq.index(dataset.char_to_idx['<EOS>'])
                    true_seq = true_seq[:eos_pos]
                true_text = "".join([dataset.idx_to_char[idx] for idx in true_seq]).lower()
                if pred_text == true_text:
                    seq_correct += 1
                seq_total += 1
                min_len = min(len(pred_text), len(true_text))
                match_cnt = sum(1 for j in range(min_len) if pred_text[j] == true_text[j])
                char_correct += match_cnt
                char_total += max(len(pred_text), len(true_text))
    seq_acc = seq_correct / seq_total if seq_total > 0 else 0.0
    char_acc = char_correct / char_total if char_total > 0 else 0.0
    print(f"Validation: Full Sequence Acc: {seq_acc*100:.2f}%, Character Acc: {char_acc*100:.2f}%")
    
    sample_loader = DataLoader(val_dataset, batch_size=5, shuffle=True, collate_fn=captcha_collate_fn)
    sample_images, sample_dec_input, sample_dec_target = next(iter(sample_loader))
    sample_images = sample_images.to(device)
    logits = model(sample_images) 
    log_probs = logits.log_softmax(2)
    _, sample_pred_indices = torch.max(log_probs, dim=2)
    sample_pred_indices = sample_pred_indices.permute(1, 0)
    sample_preds = []
    for i in range(sample_pred_indices.size(0)):
        pred_seq = sample_pred_indices[i].tolist()
        pred_text = ""
        prev = None
        for idx in pred_seq:
            if idx != dataset.char_to_idx['<PAD>'] and idx != prev:
                if idx == dataset.char_to_idx['<EOS>']:
                    break
                pred_text += dataset.idx_to_char[idx]
            prev = idx
        sample_preds.append(pred_text.lower())
    fig, axes = plt.subplots(1, len(sample_preds), figsize=(12, 3))
    if(epoch%20==19):
        for i in range(len(sample_preds)):
            img = sample_images[i].cpu().numpy().squeeze()
            true_seq = sample_dec_target[i].tolist()
            print(sample_preds)
            if dataset.char_to_idx['<EOS>'] in true_seq:
                eos_pos = true_seq.index(dataset.char_to_idx['<EOS>'])
                true_seq = true_seq[:eos_pos]
            true_text = "".join([dataset.idx_to_char[idx] for idx in true_seq]).lower()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"T: {true_text}\nP: {sample_preds[i]}", fontsize=10)
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()

test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, collate_fn=captcha_collate_fn)
images, dec_input, dec_target = next(iter(test_loader))
images = images.to(device)
logits = model(images)
log_probs = logits.log_softmax(2)
_, pred_indices = torch.max(log_probs, dim=2)
pred_indices = pred_indices.permute(1, 0)
predictions = []
for i in range(pred_indices.size(0)):
    pred_seq = pred_indices[i].tolist()
    pred_text = ""
    prev = None
    for idx in pred_seq:
        if idx != dataset.char_to_idx['<PAD>'] and idx != prev:
            if idx == dataset.char_to_idx['<EOS>']:
                break
            pred_text += dataset.idx_to_char[idx]
        prev = idx
    predictions.append(pred_text.lower())
fig, axes = plt.subplots(1, len(predictions), figsize=(12, 3))
for i in range(len(predictions)):
    img = images[i].cpu().numpy().squeeze()
    true_seq = dec_target[i].tolist()
    if dataset.char_to_idx['<EOS>'] in true_seq:
        eos_pos = true_seq.index(dataset.char_to_idx['<EOS>'])
        true_seq = true_seq[:eos_pos]
    true_text = "".join([dataset.idx_to_char[idx] for idx in true_seq]).lower()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"T: {true_text}\nP: {predictions[i]}", fontsize=10)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

if len(predictions) > 0:
    if all(seq == predictions[0] for seq in predictions):
        print(f"所有预测结果都相同: '{predictions[0]}'，模型可能陷入输出单一模式。")
    all_chars = "".join(predictions).lower()
    if all_chars:
        freq = Counter(all_chars)
        most_common_char, freq_count = freq.most_common(1)[0]
        if freq_count / len(all_chars) > 0.8:
            print(f"预测结果中字符 '{most_common_char}' 占比达到 {freq_count/len(all_chars)*100:.1f}%，模型可能偏好输出该字符。")
