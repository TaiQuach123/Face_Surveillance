import cv2 
import numpy as np
from Recognition.models import *
from Recognition.utils import load_model, cosine_score
from Recognition.data_module import create_transforms
from PIL import Image

transforms = create_transforms()

class ArcFaceRecognizer():
    def __init__(self, name="iresnet100", pretrained_path="weights/recognition/iresnet100_backbone.pth", device="cuda"):
        self._build_model(name, pretrained_path, device)
        self.model.eval()
        self.device = device
        self.preprocess = transforms['val']

    def _build_model(self, name, pretrained_path, device):
        self.cfg = None
        if name == "iresnet100":
            self.cfg = cfg_iresnet100
            self.model = iresnet100()
        elif name == "mobilefacenet":
            self.cfg = cfg_mfnet
            if self.cfg['large']:
                self.model = get_mbf_large(fp16=False, num_features=512)
            else:
                self.model = get_mbf(fp16=False, num_features=512)
        else:
            self.cfg = cfg_ghostfacenetv2
            self.model = GhostFaceNetsV2(image_size=self.cfg['image_size'], width=self.cfg['width'], dropout=self.cfg['dropout'])
        
        self.model = load_model(self.model, pretrained_path).to(device)
        self.model.eval()

    def create_embedding(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        embedding = self.model(img)

        return embedding.detach().cpu().numpy()[0]

    def verify(self, img_path1, img_path2, threshold = 0.3):
        embed1 = self.create_embedding(img_path1)
        embed2 = self.create_embedding(img_path2)

        return cosine_score(embed1, embed2)


