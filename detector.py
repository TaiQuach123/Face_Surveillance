from Detection.model import cfg_mnet, cfg_re50, RetinaFace
from Detection.utils import load_model, preprocess, PriorBox, decode, decode_landm, py_cpu_nms
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np


class RetinaFaceDetector():
    def __init__(self, name="mobilenet", pretrained_path="weights/detection/mobilenet_100.pth", device="cuda"):
        self._build_model(name, pretrained_path, device)
        self.device = device
        self.rgb_mean = (0.485, 0.456, 0.406)
        self.rgb_std = (0.229, 0.224, 0.225)

    def _build_model(self, name, pretrained_path, device):
        self.cfg = None
        if name == "mobilenet":
            self.cfg = cfg_mnet
        elif name == "resnet50":
            self.cfg = cfg_re50
        self.model = RetinaFace(cfg=self.cfg)
        self.model = load_model(self.model, pretrained_path).to(device)

        self.model.eval()
        
    def detect_single_image(self, img, nms_threshold=0.4, conf_threshold=0.5, top_k = 5000):
        resize = 1

        img_raw, img, (h, w) = preprocess(img, self.rgb_mean, self.rgb_std)
        img = img.to(self.device)

        scale_bboxes = torch.Tensor([w, h, w, h])
        scale_bboxes = scale_bboxes.to(self.device)
        scale_landms = torch.Tensor([w, h, w, h, w, h, w, h, w, h])
        scale_landms = scale_landms.to(self.device)

        loc, cls, landms = self.model(img)
        conf = F.softmax(cls, dim=-1)
        scores = conf.squeeze(0).data.cpu().numpy()[:,1]

        priorbox = PriorBox(self.cfg, image_size=(h, w))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = torch.clamp(boxes, 0, 1)
        boxes = boxes * scale_bboxes / resize
        boxes = boxes.cpu().numpy()

        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        landms = torch.clamp(landms, 0, 1)
        landms = landms * scale_landms / resize
        landms = landms.cpu().numpy()

        #ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        #keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        #do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:750, :]
        landms = landms[:750, :]
        dets = np.concatenate((dets, landms), axis=1)

        return img_raw, dets



    def extract_faces_landms(self, img_raw, dets):
        faces_bboxes = dets[:, :4].astype(int)
        scores = dets[:, 4]
        faces_landms = dets[:, 5:].reshape(-1, 5, 2) - dets[:, np.newaxis, :2]
        faces_landms = faces_landms.reshape(-1,10).astype(int)
        facial_images = []
        for face_bboxes in faces_bboxes:
            facial_image = img_raw[face_bboxes[1]:face_bboxes[3], face_bboxes[0]:face_bboxes[2], :].copy()
            
            facial_images.append(facial_image)
        
        return facial_images, faces_landms


    def align_face(self, img, left_eye, right_eye):
        angle = float(np.degrees(np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])))
        img = np.array(Image.fromarray(img).rotate(angle, resample=Image.BICUBIC))

        return img, angle





