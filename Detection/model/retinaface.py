import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models._utils as _utils
from Detection.model.commons import SSH
from torchvision.ops import FeaturePyramidNetwork


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*2, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1,1), stride=1, padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg=None):
        """
        cfg: model config
        """
        super(RetinaFace, self).__init__()
        backbone = None
        if cfg['name'] == 'resnet50':
            backbone = models.resnet50(weights='DEFAULT')
        elif cfg['name'] == 'mobilenet':
            backbone = models.mobilenet_v2(weights='DEFAULT').features

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])


        self.fpn = FeaturePyramidNetwork(in_channels_list=cfg['in_channels'], out_channels=cfg['out_channel'])
        self.ssh1 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh2 = SSH(cfg['out_channel'], cfg['out_channel'])
        self.ssh3 = SSH(cfg['out_channel'], cfg['out_channel'])


        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead
    
    def forward(self, input):
        out = self.body(input)
        fpn_out = self.fpn(out)

        feature1 = self.ssh1(fpn_out[1])
        feature2 = self.ssh2(fpn_out[2])
        feature3 = self.ssh3(fpn_out[3])

        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        return bbox_regressions, classifications, ldm_regressions



if __name__ == "__main__":
    from config import cfg_re50
    input = torch.rand((1,3,840,840))
    model = RetinaFace(cfg_re50)

    print(model(input))