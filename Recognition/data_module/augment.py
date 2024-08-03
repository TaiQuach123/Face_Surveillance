from torchvision.transforms import v2
import torch

def create_transforms(img_size=112, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    train_transforms = v2.Compose([
        v2.Resize(size=(img_size, img_size)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.RandomHorizontalFlip(0.5),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=mean, std=std)
    ])

    val_transforms = v2.Compose([
        v2.Resize(size=(img_size,img_size)),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize(mean=mean, std=std)
    ])

    return {'train': train_transforms, 'val': val_transforms}

