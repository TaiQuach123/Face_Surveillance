import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CASIADataset(Dataset):
    def __init__(self, root, txt_path, phase='train', transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.imgs_path = []
        self.labels = []
        
        self._create_data(txt_path)
    
    def _create_data(self, txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            line = line.split()
            self.imgs_path.append(os.path.join(self.root, line[0]))
            self.labels.append(int(line[1]))

    
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        img = Image.open(img_path)
        label = self.labels[idx]

        if self.transform is not None:
            if self.phase == 'train':
                transforms = self.transform['train']
            else:
                transforms = self.transform['val']
        
            img = transforms(img)

        return img, label


if __name__ == "__main__":
    dataset = CASIADataset(root='../data/CASIA-maxpy-clean')






            