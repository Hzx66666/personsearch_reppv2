from __future__ import print_function, absolute_import
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, precision_recall_curve
import os.path as osp
from tqdm import tqdm
import json
def compute_ap_cmc(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index)) 
    
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1.0
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
        # ap = ap + d_recall*precision

    return ap, cmc


def evaluate(distmat, q_pids, g_pids):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1) # from small to large

    num_no_gt = 0 # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0
    q_camids = np.zeros_like(q_pids)
    g_camids = np.ones_like(g_pids)

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids==q_pids[i])
        camera_index = np.argwhere(g_camids==q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_ap_cmc(index[i], good_index, junk_index)
        if CMC_tmp[0]==1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        print("{} query imgs do not have groundtruth.".format(num_no_gt))

    # print("R1:{}".format(num_r1))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    return CMC, mAP


def evaluate_search_psdb(q_feat, q_pids, g_feat, g_pids, g_imnames, g_ious, iou_thresh=0.5, gallery_size=100):
    fname = 'TestG{}'.format(gallery_size)
    protoc = loadmat(osp.join('/data/wangcheng/data/psdb/dataset/annotation/test/train_test', fname + '.mat'))[fname].squeeze()

    aps = []
    accs = []
    topk = [1, 5, 10]
    
    for i in tqdm(range(len(q_feat))):
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        feat_p = q_feat[i].ravel()
        probe_id = q_pids[i]
        # 1. Go through the gallery samples defined by the protocol
        for item in protoc['Gallery'][i].squeeze():
            gallery_imname = str(item[0][0])
            gallery_index = g_imnames == gallery_imname
            if gallery_index.any() == False:
                continue
            
            gallery_id = g_pids[gallery_index]
            gallery_iou = g_ious[gallery_index]
            feat_g = g_feat[gallery_index]
            gt = item[1][0].astype(np.int32)
            
            count_gt += (gt.size > 0)

            # compute distance between probe and gallery dets
            # compute cosine similarities
            sim = feat_g.dot(feat_p).ravel()
            
            # max_value = max(sim)
            # sim[sim < max_value] = sim[sim < max_value] * 0.88
            # assign label for each det
            label = []
            is_found = False
            for pid, iou in zip(gallery_id, gallery_iou):
                if not is_found and iou > iou_thresh and pid == probe_id:
                    is_found = True
                    count_tp += 1
                    label.append(1)
                else:
                    label.append(0)

            y_true.extend(list(label))
            y_score.extend(list(sim))

        # 2. Compute AP for this probe (need to scale by recall rate)
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        assert count_tp <= count_gt
        if count_gt == 0:
            recall_rate = 0
        else:
            recall_rate = count_tp * 1.0 / count_gt
        ap = 0 if count_tp == 0 else \
            average_precision_score(y_true, y_score) * recall_rate

        aps.append(ap)
        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        accs.append([min(1, sum(y_true[:k])) for k in topk])

    print('search ranking:')
    print('mAP = {:.2%}'.format(np.mean(aps)))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print('top-{:1d} = {:.2%}'.format(k, accs[i]))

    with open('search_results.txt', 'a') as f:
        f.write('search ranking:\n')
        f.write('mAP = {:.2%}\n'.format(np.mean(aps)))
        for i, k in enumerate(topk):
            f.write('top-{:1d} = {:.2%}\n'.format(k, accs[i]))
        f.write('\n')

    return accs[0]

def evaluate_search_prw(q_feat, q_pids, g_feat, g_pids, g_imnames, g_ious, iou_thresh=0.5, gallery_size=100):
    with open('/data/hzx/converted_PRW/annotations/converted_test.json','r') as gtfile:
        infos=json.load(gtfile)
        img_id_name=infos['images']
        gtbox_infos=infos['annotations']
    img_name2id={}
    for img_info in img_id_name:
        img_name2id[img_info['filename']]=img_info['id']
    pid2gtbox={}
    for ann in gtbox_infos:
        if ann['pid']==-2:
            continue
        if ann['pid'] not in pid2gtbox.keys():
            pid2gtbox[ann['pid']]={}
        pid2gtbox[ann['pid']][ann['image_id']]=ann['bbox']
    aps = []
    accs = []
    topk = [1, 5, 10]
    #with open('name.json','w') as jf:
    #    json.dump(g_imnames.tolist(),jf)
    for i in tqdm(range(len(q_feat))):
        
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        feat_p = q_feat[i].ravel()
        probe_id = q_pids[i]
        #print('pid : {}'.format(probe_id))
        # 1. Go through the gallery samples defined by the protocol
        for gallery_img in img_name2id.keys():
            gallery_imname=gallery_img
            gallery_index = g_imnames == gallery_imname
            #print(gallery_imname)
            #print(gallery_index.any())
            if gallery_index.any() == False:
                continue
            
            gallery_id = g_pids[gallery_index]
            gallery_iou = g_ious[gallery_index]
            feat_g = g_feat[gallery_index]
            if img_name2id[gallery_imname] in pid2gtbox[probe_id].keys():
                count_gt += 1

            # compute distance between probe and gallery dets
            # compute cosine similarities
            sim = feat_g.dot(feat_p).ravel()
            
            # max_value = max(sim)
            # sim[sim < max_value] = sim[sim < max_value] * 0.88
            # assign label for each det
            label = []
            is_found = False
            for pid, iou in zip(gallery_id, gallery_iou):
                if not is_found and iou > iou_thresh and pid == probe_id:
                    is_found = True
                    count_tp += 1
                    label.append(1)
                else:
                    label.append(0)

            y_true.extend(list(label))
            y_score.extend(list(sim))

        # 2. Compute AP for this probe (need to scale by recall rate)
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        assert count_tp <= count_gt
        if count_gt == 0:
            recall_rate = 0
        else:
            recall_rate = count_tp * 1.0 / count_gt
        ap = 0 if count_tp == 0 else \
            average_precision_score(y_true, y_score) * recall_rate

        aps.append(ap)
        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        accs.append([min(1, sum(y_true[:k])) for k in topk])

    print('search ranking:')
    print('mAP = {:.2%}'.format(np.mean(aps)))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print('top-{:1d} = {:.2%}'.format(k, accs[i]))

    with open('search_results.txt', 'a') as f:
        f.write('search ranking:\n')
        f.write('mAP = {:.2%}\n'.format(np.mean(aps)))
        for i, k in enumerate(topk):
            f.write('top-{:1d} = {:.2%}\n'.format(k, accs[i]))
        f.write('\n')

    return accs[0]