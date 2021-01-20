from __future__ import print_function, absolute_import
import os
import sys
import time
import math
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from dataset_loader import ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision, OIMLoss, CircleLoss, CircleLossSun, AngularPenaltySMLoss, CosFaceLoss
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate, evaluate_search_psdb, evaluate_search_prw
from samplers import RandomIdentitySampler
import pickle

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss and hard triplet loss')
# Datasets
parser.add_argument('--root', type=str, default='/data/wangcheng/data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='psdb',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split_id', type=int, default=0, help="split index")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max_epoch', default=120, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=512, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.00035, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[40, 80], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon for label smooth")
parser.add_argument('--num_instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--use_oim', action='store_true', help="use oim loss")
parser.add_argument('--angle_loss', type=str, default='softmax', choices=['softmax', 'cosface'])
parser.add_argument('--scalar', type=float, default=1.0, help="scalar for oim loss")
parser.add_argument('--momentum', type=float, default=0.0, help="momentum for oim loss")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--dropout', type=float, default=0, help="dropout for FC")
# Miscs
parser.add_argument('--distance', type=str, default='consine', help="euclidean or consine")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=5,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--person_search', action='store_true', help="use person search evaluation")
parser.add_argument('--gpu_devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_img_dataset(
        name=args.dataset, root=args.root,  train_set_type='gt', test_set_type='det'
    )

    # Data augmentation
    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability = 0.5)
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids,
                              loss={'xent', 'htri'}, dropout=args.dropout)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    if args.use_oim:
        criterion_xent = OIMLoss(num_features=2048, num_classes=dataset.num_train_pids, num_unlabeled=0, scalar=args.scalar, momentum=args.momentum)
    else:
        if args.angle_loss == 'softmax':
            criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, epsilon=args.epsilon, use_gpu=use_gpu)
        else:
            criterion_xent = AngularPenaltySMLoss(in_features=2048, out_features=dataset.num_train_pids, loss_type=args.angle_loss)
    
    criterion_htri = TripletLoss(margin=args.margin, distance=args.distance)
    
    optimizer = optim.Adam([{'params': model.parameters()}, {'params': criterion_xent.parameters(), 'lr': 1e-3}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        with torch.no_grad():
            test(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu)
        scheduler.step()
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            with torch.no_grad():
                rank1 = test(model, queryloader, galleryloader, use_gpu, person_search=args.person_search)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            if args.use_oim:
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                    'lut': criterion_xent.lut,
                    'cq': criterion_xent.cq
                }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
            else:
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, img_paths, pids, ious) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs, features = model(imgs)
        _, preds = torch.max(outputs.data, 1)
        if isinstance(outputs, tuple):
            xent_loss = DeepSupervision(criterion_xent, features, pids)
        else:
            if args.use_oim:
                xent_loss = criterion_xent(features, pids)
                _, preds = torch.max(criterion_xent.get_preds().data, 1)
            else:
                if args.angle_loss == 'softmax':
                    xent_loss = criterion_xent(outputs, pids)
                else:
                    xent_loss = criterion_xent(features, pids)
                    _, preds = torch.max(criterion_xent.get_preds().data, 1)
        
        if isinstance(features, tuple):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        # if not args.use_oim and args.angle_loss != 'softmax':
        #     htri_loss = torch.tensor(0)
        # htri_loss = torch.tensor(0)        
        loss = xent_loss + htri_loss

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'xentLoss:{xent_loss.avg:.4f} '
          'triLoss:{tri_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
           epoch+1, batch_time=batch_time,
           data_time=data_time, xent_loss=batch_xent_loss,
           tri_loss=batch_htri_loss, acc=corrects))

def fliplr(img, use_gpu):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    if use_gpu: inv_idx = inv_idx.cuda()
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], person_search=False):
    batch_time = AverageMeter()
    
    model.eval()

    qf, q_pids, q_ious = [], [], []
    for batch_idx, (imgs, img_paths, pids, ious) in enumerate(queryloader):
        end = time.time()

        n, c, h, w = imgs.size()
        features = torch.FloatTensor(n, model.module.feat_dim).zero_()
        for i in range(2):
            if(i==1):
                imgs = fliplr(imgs, use_gpu)
            if use_gpu: imgs = imgs.cuda()
            _, outputs = model(imgs)
            f = outputs.data.cpu()
            features = features+f

        batch_time.update(time.time() - end)

        qf.append(features)
        q_pids.extend(pids)
        q_ious.extend(ious)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_ious = np.asarray(q_ious)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_ious, g_imnames = [], [], [], []
    end = time.time()
    for batch_idx, (imgs, img_paths, pids, ious) in enumerate(galleryloader):
        end = time.time()

        n, c, h, w = imgs.size()
        features = torch.FloatTensor(n, model.module.feat_dim).zero_()
        for i in range(2):
            if(i==1):
                imgs = fliplr(imgs, use_gpu)
            if use_gpu: imgs = imgs.cuda()
            _, outputs = model(imgs)
            f = outputs.data.cpu()
            features = features+f

        batch_time.update(time.time() - end)

        gf.append(features)
        g_pids.extend(pids)
        g_ious.extend(ious)
        g_imnames.extend(img_paths)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_ious = np.asarray(g_ious)
    g_imnames = np.asarray(g_imnames)

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    if person_search:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        #tmp=[qf.numpy(), q_pids, gf.numpy(), g_pids, g_imnames, g_ious]
        #with open('tmpfile2.pickle','wb') as tmpf:
        #    pickle.dump(tmp,tmpf)
        if args.dataset=='psdb':
            cmc = evaluate_search_psdb(qf.numpy(), q_pids, gf.numpy(), g_pids, g_imnames, g_ious)
        elif args.dataset=='PRW':
            cmc = evaluate_search_prw(qf.numpy(), q_pids, gf.numpy(), g_pids, g_imnames, g_ious)
        else:
            print('wrong dataset!')
            cmc=[]
        return cmc
    else:
        m, n = qf.size(0), gf.size(0)
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        distmat = torch.zeros((m,n))
        if args.distance == 'euclidean':
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            for i in range(m):
                distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
        else:
            q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
            g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
            qf = qf.div(q_norm.expand_as(qf))
            gf = gf.div(g_norm.expand_as(gf))
            for i in range(m):
                distmat[i] = - torch.mm(qf[i:i+1], gf.t())
        distmat = distmat.numpy()

        # result = {'gallery_f':gf.numpy(),'gallery_label':g_pids,'gallery_cam':g_camids,'query_f':qf.numpy(),'query_label':q_pids,'query_cam':q_camids}
        # scipy.io.savemat('pytorch_result.mat',result)

        print("Computing CMC and mAP")
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=args.use_metric_cuhk03)
        cmc, mAP = evaluate(distmat, q_pids, g_pids)

        print("Results ----------")
        print("mAP: {:.2%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
        print("------------------")

        return cmc[0]

if __name__ == '__main__':
    main()
