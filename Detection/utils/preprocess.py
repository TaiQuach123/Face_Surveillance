import numpy as np
from PIL import Image
import torch
def preprocess(img, rgb_mean, rgb_std):
    #if img is a path
    if isinstance(img, str):
        img_raw = np.array(Image.open(img).convert('RGB'))
        img = np.float32(img_raw)
    else:
        img_raw = img.copy()
        img = np.float32(img_raw)
        
    h, w, _ = img.shape
    img = img/255
    img -= rgb_mean
    img = img/rgb_std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)

    return img_raw, img, (h, w)