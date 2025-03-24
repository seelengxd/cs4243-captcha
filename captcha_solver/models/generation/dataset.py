import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None, sample_size=2000):
        self.image_dir = image_dir
        all_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                     if os.path.isfile(os.path.join(image_dir, f))]
        all_names = [os.path.splitext(os.path.basename(f))[0] for f in all_files]
        chars = set()
        for name in all_names:
            chars.update(list(name))
        self.char_set = sorted(list(chars))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.char_set)}
        self.max_length = max((len(name) for name in all_names), default=0)
        self.image_files = random.sample(all_files, min(sample_size, len(all_files)))
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((64, 128)),
                transforms.RandomRotation(degrees=5, fill=0),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        filename = os.path.basename(img_path)
        text = os.path.splitext(filename)[0]
        text_onehot = torch.zeros((self.max_length, len(self.char_set)), dtype=torch.float)
        for i, ch in enumerate(text):
            if ch in self.char_to_idx:
                text_onehot[i, self.char_to_idx[ch]] = 1.0
        text_onehot = text_onehot.view(-1)
        return image, text_onehot
