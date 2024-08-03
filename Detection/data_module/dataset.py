import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class WiderFaceDataset(Dataset):
    def __init__(self, annotation_path, preproc = None):
        self.preproc = preproc
        self.imgs_path = []
        self.labels = self.process_annotations(annotation_path)

    def process_annotations(self, path):
        labels = []
        label = []
        isFirst = True

        with open(path) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                img_path = os.path.join(path.replace('label.txt', 'images'), line[2:])
                self.imgs_path.append(img_path)
                if isFirst:
                    isFirst = False  
                else:
                    labels.append(label)
                    label = []

            else:
                line = line.split()
                label.append([float(x) for x in line])
        
        labels.append(label)
        return labels



    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx]).convert('RGB')
        img = np.array(img)
        h, w, _ = img.shape
        labels = self.labels[idx]
        annotations = np.zeros((0,15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1,15))
            #bbox coordinate
            annotation[0, 0] = label[0] #x1
            annotation[0, 1] = label[1] #y1
            annotation[0, 2] = label[0] + label[2] #x2
            annotation[0, 3] = label[1] + label[3] #y2

            #landmarks
            if len(label) > 4:
                annotation[0, 4] = label[4]    # l0_x
                annotation[0, 5] = label[5]    # l0_y
                annotation[0, 6] = label[7]    # l1_x
                annotation[0, 7] = label[8]    # l1_y
                annotation[0, 8] = label[10]   # l2_x
                annotation[0, 9] = label[11]   # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
            else:
                annotation[0, 4] = -1.0    # l0_x
                annotation[0, 5] = -1.0    # l0_y
                annotation[0, 6] = -1.0    # l1_x
                annotation[0, 7] = -1.0    # l1_y
                annotation[0, 8] = -1.0   # l2_x
                annotation[0, 9] = -1.0   # l2_y
                annotation[0, 10] = -1.0  # l3_x
                annotation[0, 11] = -1.0  # l3_y
                annotation[0, 12] = -1.0  # l4_x
                annotation[0, 13] = -1.0  # l4_y
            

            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img).float(), target



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
