from __future__ import absolute_import
import sys

import torch
from torch import nn

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision', 'CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss', 'RingLoss', 'KLdivergence']

def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='euclidean', use_gpu=True):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'consine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.use_gpu = use_gpu
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'consine':
            fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            l2norm = inputs.div(fnorm.expand_as(inputs))
            dist = - torch.mm(l2norm, l2norm.t())

        if self.use_gpu: targets = targets.cuda()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self, weight_ring=1.):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.weight_ring = weight_ring

    def forward(self, x):
        l = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return l * self.weight_ring

class KLdivergence(nn.Module):
    
    def __init__(self, T=4):
        super(KLdivergence,self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.T = 4
        
    def forward(self, student, teacher):
         n, c = student.size()
         assert(student.size() == teacher.size())
         # Do not BP to teacher model
         teacher = teacher.detach()
         
         student = self.softmax(student/self.T)
         teacher = self.softmax(teacher/self.T)
         
         log_student = student.clamp(min=1e-12).log()
         log_teacher = teacher.clamp(min=1e-12).log()
         
         loss = (log_teacher - log_student) * teacher
         loss = loss.sum(dim=1, keepdim=False).mean()
         
         return loss

if __name__ == '__main__':
    pass