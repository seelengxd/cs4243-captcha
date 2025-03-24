import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None, sample_size=2000):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                             if os.path.isfile(os.path.join(image_dir, f))]
        self.image_files = self.image_files[:sample_size] if len(self.image_files) > sample_size \
                           else self.image_files

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
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_str = filename.split('_')[0] if '_' in filename else filename
        return image, label_str
