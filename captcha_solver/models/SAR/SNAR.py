import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

chars = list("0123456789abcdefghijklmnopqrstuvwxyz")
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"
chars.extend([pad_token, sos_token, eos_token])
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.transform = transform
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        filename = os.path.basename(img_path)
        label_str = filename.split('-')[0].lower()
        label_indices = [char2idx[ch] for ch in label_str]
        label_indices.append(char2idx[eos_token])
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        return image, label_indices

def pad_collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = [len(seq) for seq in labels]
    max_len = max(lengths)
    padded_labels = torch.full((len(labels), max_len), char2idx[pad_token], dtype=torch.long)
    for i, seq in enumerate(labels):
        padded_labels[i, :len(seq)] = seq
    return images, padded_labels, lengths

transform = transforms.Compose([
    transforms.Resize((79, 729)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=(2,1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=(2,1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
    def forward(self, x):
        features = self.conv_block(x)
        features = self.pool(features)
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)
        return features

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        return outputs, (h_n, c_n)

class NonAutoregressiveDecoder(nn.Module):
    def __init__(self, hidden_size, enc_output_dim, vocab_size, embedding_dim=128, attention_dim=128, dropout=0.5, max_len=10):
        super(NonAutoregressiveDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.enc_output_dim = enc_output_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pos_embedding = nn.Parameter(torch.randn(max_len, embedding_dim))
        self.dropout = nn.Dropout(dropout)
        self.dec_proj = nn.Linear(embedding_dim, attention_dim)
        self.enc_proj = nn.Linear(enc_output_dim, attention_dim)
        self.attn_score = nn.Linear(attention_dim, 1)
        self.hidden_proj = nn.Linear(embedding_dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size + enc_output_dim, vocab_size)
    
    def forward(self, encoder_outputs, target_seq=None):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        enc_proj = self.enc_proj(encoder_outputs)
        if target_seq is not None:
            max_steps = target_seq.size(1)
        else:
            max_steps = self.max_len
        pos_emb = self.pos_embedding[:max_steps].unsqueeze(0).expand(batch_size, max_steps, -1)
        pos_emb = self.dropout(pos_emb)
        dec_query = self.dec_proj(pos_emb)
        combined = torch.tanh(enc_proj.unsqueeze(1) + dec_query.unsqueeze(2))
        attn_scores = self.attn_score(combined).squeeze(3)
        attn_weights = torch.softmax(attn_scores, dim=2)
        context = torch.bmm(attn_weights, encoder_outputs)
        hidden = self.hidden_proj(pos_emb)
        hidden = self.dropout(hidden)
        logits = self.fc_out(torch.cat([hidden, context], dim=-1))
        if target_seq is None:
            preds = logits.argmax(dim=2)
            return logits, preds
        return logits

class SARModel(nn.Module):
    def __init__(self, vocab_size, encoder_hidden=256, decoder_hidden=256, embedding_dim=128, attention_dim=128, dec_max_len=10):
        super(SARModel, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.encoder = EncoderRNN(input_size=256, hidden_size=encoder_hidden, num_layers=1, dropout=0.5)
        enc_output_dim = 2 * encoder_hidden
        self.decoder = NonAutoregressiveDecoder(hidden_size=decoder_hidden, enc_output_dim=enc_output_dim, vocab_size=vocab_size, embedding_dim=embedding_dim, attention_dim=attention_dim, dropout=0.5, max_len=dec_max_len)
    
    def forward(self, images, targets=None):
        features = self.cnn(images)
        enc_outputs, _ = self.encoder(features)
        if targets is not None:
            outputs = self.decoder(enc_outputs, target_seq=targets)
        else:
            outputs, preds = self.decoder(enc_outputs, target_seq=None)
            return outputs, preds
        return outputs

def decode_predictions(pred_tensor):
    pred_strings = []
    for seq in pred_tensor:
        s = ""
        for idx in seq.tolist():
            ch = idx2char.get(idx, '')
            if ch == eos_token:
                break
            if ch in [sos_token, pad_token]:
                continue
            s += ch
        pred_strings.append(s)
    return pred_strings

def denormalize_image(tensor_img):
    img = tensor_img.cpu().clone()
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1)
    np_img = img.permute(1, 2, 0).numpy()
    return np_img

def show_images(loader, model, device, num_images=4, title_prefix=""):
    model.eval()
    with torch.no_grad():
        images, padded_targets, _ = next(iter(loader))
        images = images.to(device)
        _, preds = model(images, targets=None)
        pred_strs = decode_predictions(preds)
        gt_strs = []
        for seq in padded_targets:
            s = ""
            for idx in seq.tolist():
                ch = idx2char.get(idx, '')
                if ch == eos_token:
                    break
                if ch in [sos_token, pad_token]:
                    continue
                s += ch
            gt_strs.append(s)
    plt.figure(figsize=(12, 6))
    for i in range(min(num_images, images.size(0))):
        plt.subplot(1, num_images, i+1)
        plt.imshow(denormalize_image(images[i]))
        plt.title(f"{title_prefix}\nGT: {gt_strs[i]}\nPred: {pred_strs[i]}")
        plt.axis("off")
    plt.show()
    model.train()

def test_model(test_loader, model, device, model_path="sar_captcha_best_nar.pth"):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_gt = []
    all_preds = []
    total_char_correct = 0
    total_char_count = 0
    total_word_correct = 0
    total_word_count = 0
    with torch.no_grad():
        for images, padded_targets, _ in test_loader:
            images = images.to(device)
            padded_targets = padded_targets.to(device)
            _, preds = model(images, targets=None)
            pred_strs = decode_predictions(preds)
            gt_strs = []
            for seq in padded_targets:
                s = ""
                for idx in seq.tolist():
                    ch = idx2char.get(idx, '')
                    if ch == eos_token:
                        break
                    if ch in [sos_token, pad_token]:
                        continue
                    s += ch
                gt_strs.append(s)
            for gt, pred in zip(gt_strs, pred_strs):
                all_gt.append(gt)
                all_preds.append(pred)
                total_char_correct += sum(1 for g, p in zip(gt, pred) if g == p)
                total_char_count += len(gt)
                if gt == pred:
                    total_word_correct += 1
                total_word_count += 1
    char_match_rate = total_char_correct / total_char_count if total_char_count > 0 else 0
    word_match_rate = total_word_correct / total_word_count if total_word_count > 0 else 0
    print(f"Single Character Match Rate: {char_match_rate:.4f}")
    print(f"Complete Word Match Rate: {word_match_rate:.4f}")
    show_images(test_loader, model, device, num_images=6, title_prefix="Test Results")
    return all_gt, all_preds, char_match_rate, word_match_rate

def main():
    train_dir = "cs4243-captcha-main\\cs4243-captcha-main\\captcha_solver\\data\\train_cleaned_color_resized"
    test_dir = "cs4243-captcha-main\\cs4243-captcha-main\\captcha_solver\\data\\test_cleaned_color_resized"
    full_dataset = CaptchaDataset(train_dir, transform=transform)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    test_dataset = CaptchaDataset(test_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    vocab_size = len(char2idx)
    model = SARModel(vocab_size=vocab_size, encoder_hidden=256, decoder_hidden=256, embedding_dim=128, attention_dim=128, dec_max_len=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[pad_token])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 100
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        model.train()
        total_train_loss = 0.0
        for images, padded_targets, _ in train_loader:
            images = images.to(device)
            padded_targets = padded_targets.to(device)
            optimizer.zero_grad()
            logits = model(images, targets=padded_targets)
            B, T, V = logits.size()
            loss = criterion(logits.view(B * T, V), padded_targets.view(B * T))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, padded_targets, _ in val_loader:
                images = images.to(device)
                padded_targets = padded_targets.to(device)
                logits = model(images, targets=padded_targets)
                B, T, V = logits.size()
                loss = criterion(logits.view(B * T, V), padded_targets.view(B * T))
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "sar_captcha_best_nar.pth")
            print("Best model saved.")
    print("Testing on test set:")
    test_model(test_loader, model, device, model_path="sar_captcha_best_nar.pth")

if __name__ == "__main__":
    main()
