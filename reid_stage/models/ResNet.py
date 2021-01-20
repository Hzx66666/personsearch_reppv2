from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torchvision

from .resnets1 import resnet50

__all__ = ['ResNet50', 'ResNet101', 'ResNet50M']


# class ResNet50(nn.Module):
#     def __init__(self, num_classes, loss={'xent'}, dropout=0, **kwargs):
#         super(ResNet50, self).__init__()
#         self.loss = loss
#         resnet50 = torchvision.models.resnet50(pretrained=True)
#         self.base = nn.Sequential(*list(resnet50.children())[:-2])

#         num_ftrs = resnet50.fc.in_features
#         self.classifier = nn.Linear(num_ftrs, num_classes)
#         # self.classifier.apply(weights_init_classifier)

#         self.feat_dim = 2048 # feature dimension

#     def forward(self, x):
#         x = self.base(x)
#         x = F.avg_pool2d(x, x.size()[2:])
#         f = x.view(x.size(0), -1)
#         if not self.training:
#             return f
            
#         y = self.classifier(f)

#         if self.loss == {'xent'}:
#             return y
#         elif self.loss == {'xent', 'htri'}:
#             return y, f
#         elif self.loss == {'cent'}:
#             return y, f
#         elif self.loss == {'ring'}:
#             return y, f
#         else:
#             raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, dropout=0, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50_ft = resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50_ft.children())[:-2])

        num_ftrs = resnet50_ft.fc.in_features

        # add_block = []
        # add_block += [nn.BatchNorm1d(num_ftrs)]
        # add_block += [nn.LeakyReLU(0.1)]
        # add_block += [nn.Dropout(p=dropout)]
        # add_block = nn.Sequential(*add_block)
        # self.bn = add_block
        self.bn = nn.BatchNorm1d(num_ftrs)
        # self.relu = nn.LeakyReLU(0.1)
        # self.dropout = nn.Dropout(p=dropout)

        self.classifier = nn.Linear(num_ftrs, num_classes)

        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
            
        f = self.bn(x)
        y = self.classifier(f)

        return y, f



class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        elif self.loss == {'cent'}:
            return y, f
        elif self.loss == {'ring'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50M(nn.Module):
    """ResNet50 + mid-level features.

    Reference:
    Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
    Cross-Domain Instance Matching. arXiv:1711.08106.
    """
    def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
        super(ResNet50M, self).__init__()
        self.loss = loss
        resnet50_ft = resnet50(pretrained=True)
        # resnet50 = torchvision.models.resnet50(pretrained=True)
        base = nn.Sequential(*list(resnet50_ft.children())[:-2])
        self.layers1 = nn.Sequential(base[0], base[1], base[2])
        self.layers2 = nn.Sequential(base[3], base[4])
        self.layers3 = base[5]
        self.layers4 = base[6]
        self.layers5a = base[7][0]
        self.layers5b = base[7][1]
        self.layers5c = base[7][2]
        self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.classifier = nn.Linear(3072, num_classes)
        self.bn = nn.BatchNorm1d(2048)
        self.feat_dim = 3072 # feature dimension

    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5a = self.layers5a(x4)
        x5b = self.layers5b(x5a)
        x5c = self.layers5c(x5b)

        x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))
        x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))
        x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))

        midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)
        midfeat = self.fc_fuse(midfeat)

        x5c_feat = self.bn(x5c_feat)
        combofeat = torch.cat((x5c_feat, midfeat), dim=1)
        if not self.training:
            return combofeat
        prelogits = self.classifier(combofeat)
        
        if self.loss == {'xent'}:
            return prelogits
        elif self.loss == {'xent', 'htri'}:
            return prelogits, combofeat
        elif self.loss == {'cent'}:
            return prelogits, combofeat
        elif self.loss == {'ring'}:
            return prelogits, combofeat
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))