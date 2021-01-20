from __future__ import absolute_import
import sys

import torch
from torch import nn
from torch import autograd
"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision', 'CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss', 'RingLoss', 'KLdivergence', 'OIMLoss', 'CircleLoss', 'CircleLossSun', 'AngularPenaltySMLoss', 'CosFaceLoss']


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


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
        # unlabeled_index = targets<0
        # targets[unlabeled_index] = 0
        # import ipdb; ipdb.set_trace()
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu().long(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # targets[unlabeled_index] = 0
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

class OIM(autograd.Function):
    def __init__(self, lut, cq, momentum=0.0):
        super(OIM, self).__init__()
        self.lut = lut
        self.cq = cq
        self.momentum = momentum
        self.header = 0

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.cq.t())

        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([self.lut, self.cq], dim=0))
        
        id_to_feat = {}
        for x, y in zip(inputs, targets):
            if y.item() in id_to_feat:
                id_to_feat[y.item()]['counter'] += 1
                id_to_feat[y.item()]['feat'] += x
            else:
                id_to_feat[y.item()] = {}
                id_to_feat[y.item()]['counter'] = 1
                id_to_feat[y.item()]['feat'] = x

        for key in id_to_feat.keys():
            self.lut[key] = self.momentum * self.lut[key] + (1. - self.momentum) * id_to_feat[key]['feat'] / id_to_feat[key]['counter']
            self.lut[key] /= self.lut[key].norm()

        return grad_inputs, None


def oim(inputs, targets, lut, cq, momentum=0.0):
    return OIM(lut, cq, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, num_unlabeled=1000, scalar=50.0, momentum=0.0, weight=None, size_average=True, use_oim_pkl=False, lut_file=None, cq_file=None):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_unlabeled = num_unlabeled
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut', torch.rand(num_classes, num_features))
        self.register_buffer('cq',  torch.rand(num_unlabeled, num_features))
        self.lut = F.normalize(self.lut, p=2, dim=1)
        self.cq = F.normalize(self.cq, p=2, dim=1)
 
        if use_oim_pkl:
            if os.path.exists(lut_file):
                f1 = open(lut_file, 'rb')
                self.lut = pickle.loads(f1.read())
                f1.close()
            if os.path.exists(cq_file):
                f2 = open(cq_file, 'rb')
                self.cq = pickle.loads(f2.read())
                f2.close()
        
        self.lut = self.lut[:num_classes, :]
        self.lut = self.lut.cuda()
        self.cq = self.cq[:num_unlabeled, :]
        self.cq = self.cq.cuda()
        self.ce_loss = CrossEntropyLabelSmooth(num_classes)

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, p=2, dim=1)
        inputs = oim(inputs, targets, self.lut, self.cq, momentum=self.momentum)
        inputs *= self.scalar
        self.pred = inputs

        # loss = self.ce_loss(inputs, targets)
        
        if self.size_average:
            loss = F.cross_entropy(inputs, targets.long(), reduction='mean', ignore_index=-1)
        else:
            loss = F.cross_entropy(inputs, targets.long(), ignore_index=-1)

        return loss
    
    def get_preds(self):
        return self.pred
        

class CircleLoss(nn.Module):
    def __init__(self, margin=0.25, alpha=128):
        super(CircleLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
    
    def forward(self, embedding, all_targets):
        all_embedding = F.normalize(embedding, dim=1)
        dist_mat = torch.matmul(all_embedding, all_embedding.t())

        N = dist_mat.size(0)
        is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t()).float()

        # Compute the mask which ignores the relevance score of the query to itself
        is_pos = is_pos - torch.eye(N, N, device=is_pos.device)
        is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.margin, min=0.)
        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - self.alpha * alpha_p * (s_p - delta_p)
        logit_n = self.alpha * alpha_n * (s_n - delta_n)

        loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss


class CircleLossSun(nn.Module):
    def __init__(self, m=0.35, gamma=96):
        super(CircleLossSun, self).__init__()
        self.m = m
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        n = targets.size(0)
        feat = inputs / torch.sqrt((inputs ** 2).sum(axis=1, keepdim=True) + 1e-9)
        sim = torch.matmul(feat, feat.t())
        mask = targets.expand(n, n).eq(targets.expand(n, n).t()).float()
        pos_scale = torch.detach(self.gamma * (1 + self.m - sim))
        neg_scale = torch.detach(self.gamma * F.relu(sim + self.m))
        scale_matrix = pos_scale * mask + neg_scale * (1 - mask)

        simi = (sim - mask * (1 - self.m) - (1 - mask) * self.m) * scale_matrix
        neg_sim_sum = (torch.exp(simi) * (1 - mask)).sum(dim=1)
        pos_sim_sum = (torch.exp(simi) * mask).sum(dim=1)

        loss = torch.log(1 + neg_sim_sum / pos_sim_sum).mean()

        return loss


class CosFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-7, s=None, m=None):
        '''
        CosFace Loss
        
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(CosFaceLoss, self).__init__()
        self.s = 30.0 if not s else s
        self.m = 0.4 if not m else m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.fc = self.fc.cuda()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        self.pred = wf
        labels = labels.long()

        mask = torch.zeros(wf.size()).cuda().scatter_(1, labels.unsqueeze(1).data.long(), 1)
        mask = mask * self.m
        wf_ams = wf - mask

        loss = F.cross_entropy(wf_ams, labels, reduction='mean', ignore_index=-1)

        return loss

    def get_preds(self):
        return self.pred


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.fc = self.fc.cuda()
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        self.pred = wf
        labels = labels.long()
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
    
    def get_preds(self):
        return self.pred


if __name__ == '__main__':
    pass