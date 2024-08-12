from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from Detection.model import cfg_mnet, cfg_re50
from Detection.utils import PriorBox, py_cpu_nms, decode, decode_landm, load_model
import cv2
from Detection.model import RetinaFace
import time
from PIL import Image

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='weights/detection/mobilenet_100.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobilenet', help='Backbone network mobilenet or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--img_path', default='test_imgs/0.jpg', type=str, help='img path for detection')
parser.add_argument('--save_result_path', default='test_imgs/3_result.jpg', type=str, help='path to store detection result')

args = parser.parse_args()

cudnn.benchmark = True
torch.set_grad_enabled(False)

cfg = None
if args.network == "mobilenet":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (0.485, 0.456, 0.406) 
rgb_std = (0.229, 0.224, 0.225)

net = RetinaFace(cfg=cfg)
net = load_model(net, args.trained_model, args.cpu)
net.eval()

device = torch.device("cpu" if args.cpu else "cuda")
net = net.to(device)

#net = torch.compile(net)

def detect(net, img_path, device):
    resize = 1

    img_raw = np.array(Image.open(img_path).convert('RGB'))
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img = img/255
    img -= rgb_mean
    img = img/rgb_std
    img = img.transpose(2, 0, 1)
    
    img = torch.from_numpy(img).float().unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    for i in range(10):
        tic = time.time()
        loc, cls, landms = net(img)  # forward pass
        conf = F.softmax(cls, dim=-1)
        print('net forward time: {:.4f}'.format(time.time() - tic))

    tic = time.time()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = torch.clamp(boxes, 0, 1)
    boxes = boxes * scale / 1
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    landms = torch.clamp(landms, 0, 1)
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()


    # ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:750, :]
    landms = landms[:750, :]
    dets = np.concatenate((dets, landms), axis=1)

    print('Total: ', time.time() - tic)
    # show image
    if True:
        for b in dets:
            if b[4] < 0.6:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image

        name = args.save_result_path
        cv2.imwrite(name, cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    detect(net, args.img_path, device)
    