import cv2 
import os
import numpy as np
from Recognition.models import *
from Recognition.utils import load_model, cosine_score
from Recognition.data_module import create_transforms
from PIL import Image
import pickle

transforms = create_transforms()

class ArcFaceRecognizer():
    def __init__(self, name="iresnet100", pretrained_path="weights/recognition/iresnet100_backbone.pth", device="cuda", face_db_file = None):
        self._build_model(name, pretrained_path, device)
        self.model.eval()
        self.device = device
        self.preprocess = transforms['val']
        #self._create_face_database(folder)
        
        if face_db_file is not None:
            with open(face_db_file, 'rb') as f:
                self.face_db = pickle.load(f)

            embeddings = []
            mapping = {}

            for i, person in enumerate(self.face_db.keys()):
                embeddings.append(self.face_db[person])
                mapping[i] = person
            
            self.embeddings = np.array(embeddings)
            self.mapping = mapping
            
        else:
            self.face_db = None
            self.embeddings = None
            self.mapping = None
        
    
    def _create_face_database(self, folder, save_file):
        #folder
        #|---person1
        #|   |---aligned.jpg
        #|---person2
        #    |---aligned.jpg 

        dct = {}

        for person in os.listdir(folder):
            path = os.path.join(folder, person)
            dct[person] = os.path.join(path, 'aligned.jpg')
        
        face_db = {}
        mapping = {}
        embeddings = []
        for i, person in enumerate(dct.keys()):
            img_path = dct[person]
            emb = self.create_embedding(img_path)

            face_db[person] = emb
            mapping[i] = person
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)

        self.face_db = face_db
        self.mapping = mapping
        self.embeddings = embeddings

        with open(save_file, 'wb') as f:
            pickle.dump(self.face_db, f)


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

    def create_embedding(self, img):
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img, 'RGB')
        
        img = self.preprocess(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        embedding = self.model(img)

        return embedding.detach().cpu().numpy()[0]


    def find_person(self, img, threshold = 0.25):
        assert self.face_db is not None, "No Database Available"
        emb = self.create_embedding(img)
        scores = (self.embeddings @ emb.reshape(-1)) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(emb))

        idx = np.argmax(scores)

        if scores[idx] >= threshold:
            return self.mapping[idx], scores[idx]
        else:
            return 'unknown', scores[idx]


    def verify(self, img_path1, img_path2, threshold = 0.3):
        embed1 = self.create_embedding(img_path1)
        embed2 = self.create_embedding(img_path2)
        score = cosine_score(embed1, embed2)

        result = {'verify': score > threshold, 'cosine_score': score, 'threshold': threshold}
        return result


