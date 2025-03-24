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

class Seq2SeqOCRModel(nn.Module):
    def __init__(self, vocab_size, enc_channels=64, emb_dim=64, hidden_dim=128):
        super(Seq2SeqOCRModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(32, enc_channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.decoder_cell = nn.LSTMCell(emb_dim + enc_channels, hidden_dim)
        self.attn_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.attn_enc = nn.Linear(enc_channels, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(hidden_dim + enc_channels, vocab_size)
        self.char_to_idx = None
        self.idx_to_char = None

    def forward(self, images, dec_input=None, max_len=20):
        feats = self.encoder(images)   
        B, C, Hc, Wc = feats.shape
        feats = feats.view(B, C, -1).permute(0, 2, 1)
        seq_len = feats.size(1)
        h = torch.zeros(B, self.decoder_cell.hidden_size, device=images.device)
        c = torch.zeros(B, self.decoder_cell.hidden_size, device=images.device)
        outputs = []
        if dec_input is not None:
            L = dec_input.size(1)
            for t in range(L):
                emb = self.embedding(dec_input[:, t]) 
                Wh = self.attn_hidden(h).unsqueeze(1).expand(-1, seq_len, -1)  
                Ue = self.attn_enc(feats)                                  
                attn_scores = torch.tanh(Wh + Ue)                           
                attn_scores = self.attn_score(attn_scores).squeeze(-1)        
                attn_weights = torch.softmax(attn_scores, dim=1)            
                context = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)  
                h, c = self.decoder_cell(torch.cat([emb, context], dim=1), (h, c))
                logits = self.out(torch.cat([h, context], dim=1))  
                outputs.append(logits)
            outputs = torch.stack(outputs, dim=1)  
            return outputs
        else:
            assert self.char_to_idx is not None and self.idx_to_char is not None, "需要提供字符映射进行解码"
            sos_idx = self.char_to_idx['<SOS>']
            eos_idx = self.char_to_idx['<EOS>']
            input_idx = torch.tensor([sos_idx] * B, device=images.device)
            generated = [''] * B
            finished = [False] * B
            for t in range(max_len):
                emb = self.embedding(input_idx) 
                Wh = self.attn_hidden(h).unsqueeze(1).expand(-1, seq_len, -1)
                Ue = self.attn_enc(feats)
                attn_scores = torch.tanh(Wh + Ue)
                attn_scores = self.attn_score(attn_scores).squeeze(-1)
                attn_weights = torch.softmax(attn_scores, dim=1)
                context = torch.bmm(attn_weights.unsqueeze(1), feats).squeeze(1)
                h, c = self.decoder_cell(torch.cat([emb, context], dim=1), (h, c))
                logits = self.out(torch.cat([h, context], dim=1))  
                _, top_idx = torch.max(logits, dim=1) 
                for i in range(B):
                    if not finished[i]:
                        idx = top_idx[i].item()
                        if idx == eos_idx:
                            finished[i] = True
                        else:
                            generated[i] += self.idx_to_char.get(idx, '').lower()
                if all(finished):
                    break
                next_input = [eos_idx if finished[i] else top_idx[i].item() for i in range(B)]
                input_idx = torch.tensor(next_input, device=images.device)
            return generated

model = Seq2SeqOCRModel(vocab_size=dataset.vocab_size)
model.char_to_idx = dataset.char_to_idx
model.idx_to_char = dataset.idx_to_char
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_idx['<PAD>'], reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 1000
train_losses = []
val_losses = []
val_seq_accuracies = []
val_char_correct = 0
val_char_total = 0

for epoch in range(num_epochs):
    model.train()
    total_loss_sum = 0.0
    total_tokens = 0
    for images, dec_input, dec_target in train_loader:
        images, dec_input, dec_target = images.to(device), dec_input.to(device), dec_target.to(device)
        optimizer.zero_grad()
        outputs = model(images, dec_input) 
        B, L, V = outputs.size()
        outputs_flat = outputs.view(B * L, V)
        targets_flat = dec_target.view(B * L)
        loss_sum = criterion(outputs_flat, targets_flat)
        non_pad = (targets_flat != dataset.char_to_idx['<PAD>'])
        num_tokens = non_pad.sum().item()
        loss = loss_sum / (num_tokens if num_tokens > 0 else 1)
        loss.backward()
        optimizer.step()
        total_loss_sum += loss_sum.item()
        total_tokens += num_tokens
    avg_train_loss = total_loss_sum / (total_tokens if total_tokens > 0 else 1)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss_sum = 0.0
    val_tokens = 0
    seq_correct = 0   
    seq_total = 0
    char_correct = 0  
    char_total = 0
    with torch.no_grad():
        for images, dec_input, dec_target in val_loader:
            images, dec_input, dec_target = images.to(device), dec_input.to(device), dec_target.to(device)
            outputs = model(images, dec_input)
            B, L, V = outputs.shape
            outputs_flat = outputs.view(B * L, V)
            targets_flat = dec_target.view(B * L)
            vloss_sum = criterion(outputs_flat, targets_flat)
            non_pad = (targets_flat != dataset.char_to_idx['<PAD>'])
            val_loss_sum += vloss_sum.item()
            val_tokens += non_pad.sum().item()
            
            pred_seq_batch = model(images) 
            for i in range(B):
                target_seq = dec_target[i].tolist()
                eos_idx = dataset.char_to_idx['<EOS>']
                pad_idx = dataset.char_to_idx['<PAD>']
                eos_pos = target_seq.index(eos_idx) if eos_idx in target_seq else len(target_seq)
                true_indices = [idx for idx in target_seq[:eos_pos] if idx not in (pad_idx, eos_idx)]
                true_label = ''.join([dataset.idx_to_char[idx] for idx in true_indices]).lower()
                pred_label = pred_seq_batch[i].lower()
                if pred_label == true_label:
                    seq_correct += 1
                seq_total += 1
                min_len = min(len(pred_label), len(true_label))
                for j in range(min_len):
                    if pred_label[j] == true_label[j]:
                        char_correct += 1
                char_total += max(len(pred_label), len(true_label))
    avg_val_loss = val_loss_sum / (val_tokens if val_tokens > 0 else 1)
    val_losses.append(avg_val_loss)
    seq_accuracy = seq_correct / seq_total if seq_total > 0 else 0.0
    char_accuracy = char_correct / char_total if char_total > 0 else 0.0
    val_seq_accuracies.append(seq_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Seq Acc: {seq_accuracy*100:.2f}%, Char Acc: {char_accuracy*100:.2f}%")

    if(epoch%20==19):
        sample_loader = DataLoader(val_dataset, batch_size=5, shuffle=True, collate_fn=captcha_collate_fn)
        sample_images, sample_dec_input, sample_dec_target = next(iter(sample_loader))
        sample_images = sample_images.to(device)
        sample_preds = model(sample_images)  
        fig, axes = plt.subplots(1, len(sample_preds), figsize=(12, 3))
        for i in range(len(sample_preds)):
            img = sample_images[i].cpu().numpy().squeeze()
            target_seq = sample_dec_target[i].tolist()
            eos_idx = dataset.char_to_idx['<EOS>']
            pad_idx = dataset.char_to_idx['<PAD>']
            eos_pos = target_seq.index(eos_idx) if eos_idx in target_seq else len(target_seq)
            true_indices = [idx for idx in target_seq[:eos_pos] if idx not in (pad_idx, eos_idx)]
            true_label = ''.join([dataset.idx_to_char[idx] for idx in true_indices]).lower()
            pred_label = sample_preds[i].lower()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"T: {true_label}\nP: {pred_label}", fontsize=10)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True, collate_fn=captcha_collate_fn)
images, dec_input, dec_target = next(iter(test_loader))
images = images.to(device)
pred_sequences = model(images) 
fig, axes = plt.subplots(1, len(pred_sequences), figsize=(12, 3))
for i in range(len(pred_sequences)):
    img = images[i].cpu().numpy().squeeze()
    true_seq = dec_target[i].tolist()
    eos_idx = dataset.char_to_idx['<EOS>']
    pad_idx = dataset.char_to_idx['<PAD>']
    eos_pos = true_seq.index(eos_idx) if eos_idx in true_seq else len(true_seq)
    true_indices = [idx for idx in true_seq[:eos_pos] if idx not in (pad_idx, eos_idx)]
    true_label = ''.join([dataset.idx_to_char[idx] for idx in true_indices]).lower()
    pred_label = pred_sequences[i].lower()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"T: {true_label}\nP: {pred_label}", fontsize=10)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

if len(pred_sequences) > 0:
    if all(seq == pred_sequences[0] for seq in pred_sequences):
        print(f"所有预测结果都相同: '{pred_sequences[0]}'，模型可能陷入输出单一模式。")
    all_chars = ''.join(pred_sequences).lower()
    if all_chars:
        freq = Counter(all_chars)
        most_common_char, freq_count = freq.most_common(1)[0]
        if freq_count / len(all_chars) > 0.8:
            print(f"预测结果中字符 '{most_common_char}' 占比达到 {freq_count/len(all_chars)*100:.1f}%，模型可能偏好输出该字符。")
