from __future__ import print_function
import os
import torch
import numpy as np
import time
from PIL import Image
from torch.nn import DataParallel
from Recognition.models import GhostFaceNetsV2
from collections import OrderedDict

def load_model(model, state_dict, load_to_cpu=False):
    new_state_dict = OrderedDict()
    if not isinstance(state_dict, OrderedDict):
        if load_to_cpu:
            state_dict = torch.load(state_dict, weights_only=True, map_location='cpu')
        else:
            state_dict = torch.load(state_dict, weights_only=True)
        
    for k in state_dict.keys():
        if k.startswith('module.'):
            new_k = k[7:]
            new_state_dict[new_k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    
    model.load_state_dict(new_state_dict)
    return model


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def get_features(model, transforms, test_list, batch_size=64):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = Image.open(img_path)
        image = transforms['val'](image).unsqueeze(0)

        if images is None:
            images = image
        else:
            images = torch.concat((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1
            images = images.to(torch.device("cuda"))
            output = model(images)
            feature = output.data.cpu().numpy()

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    
    return fe_dict

def cosine_score(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) *  np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th
        
    return best_acc, best_th


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as f:
        pairs = f.readlines()
    
    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosine_score(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)
    
    acc, th = cal_accuracy(sims, labels)
    return acc, th

def lfw_test(model, transforms, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_features(model, transforms, img_paths, batch_size=batch_size)

    t = time.time() - s
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    return acc, th, t