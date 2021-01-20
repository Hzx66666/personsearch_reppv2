from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
import pickle
from imageio import imsave

from utils import mkdir_if_missing, write_json, read_json

"""Image ReID"""

class PSDB(object):
    """
    PSDB
    """
    dataset_dir = 'psdb'
    img_dir = 'dataset/Image/SSM'

    def __init__(self, root='data', train_set_type='gt', test_set_type='det'):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, self.img_dir)
        self.train_pkl = osp.join(self.dataset_dir, 'pre_train.pkl')
        self.gallery_pkl = osp.join(self.dataset_dir, 'pre_test.pkl')
        self.query_mat = osp.join(self.dataset_dir, 'TestG50.mat')
        self.query_pid_to_img = {}

        self._check_before_run()
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_pkl, train_set_type)
        query, num_query_pids, num_query_imgs = self._process_query(self.query_mat)
        
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_pkl, test_set_type, use_unlabeled=True)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> psdb loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        print("test_type: ",test_set_type)
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_pkl):
            raise RuntimeError("'{}' is not available".format(self.train_pkl))
        if not osp.exists(self.gallery_pkl):
            raise RuntimeError("'{}' is not available".format(self.gallery_pkl))

    def _process_dir(self, pkl_path, type='gt', use_unlabeled=False):
        with open(pkl_path, 'rb') as f:
            det_results = pickle.load(f)
        
        dataset = []
        pids = set()
        num_imgs = 0
        
        for img_data in det_results:
            img_path = os.path.join(self.img_dir, img_data['im_name'])
            if type == 'det' or type == 'mixed':
                for i in range(img_data['det_box_num']):
                    box = img_data['det_boxes'][i]
                    pid = img_data['det_pids'][i]
                    if pid == -1 and not use_unlabeled:
                        continue
                    iou = img_data['det_ious'][i]
                    score = img_data['det_scores'][i]
                    
                    if self.query_pid_to_img != {} and pid != -1 and self.query_pid_to_img[pid] == img_path:
                        continue
                    dataset.append((img_path, box, pid, iou))
                    pids.add(pid)
                    # num_imgs represent the number of labeled instance
                    num_imgs += 1

            if type != 'det':
                for i in range(img_data['gt_box_num']):
                    box = img_data['gt_boxes'][i]
                    pid = img_data['gt_pids'][i]
                    if pid == -1:
                        continue
                    iou = 1.0

                    if self.query_pid_to_img != {} and self.query_pid_to_img[pid] == img_path:
                        continue
                    dataset.append((img_path, box, pid, iou))
                    pids.add(pid)
                    num_imgs += 1
        
        # pids.remove(-1)
        return dataset, len(pids), num_imgs
    
    def _process_query(self, query_mat):
        dataset = []
        test = loadmat(osp.join(self.dataset_dir,
                                    'dataset/annotation/test/train_test/TestG50.mat'))
        test = test['TestG50'].squeeze()
        for index, item in enumerate(test):
            # query
            im_name = os.path.join(self.img_dir, str(item['Query'][0,0][0][0]))
            box = item['Query'][0,0][1].squeeze().astype(np.int32)
            box[2] += box[0]
            box[3] += box[1]
            dataset.append((im_name, box, index, 1.0))
            self.query_pid_to_img[index] = im_name

        num_pids = len(test)
        num_imgs = len(test)
        return dataset, num_pids, num_imgs

class PRW(object):
    """
    PRW
    """
    dataset_dir = 'PRW'
    img_dir = 'frames'
    det_result_dir = '/data/hzx/det_result_for_tcts/prw_result_pkl'
    def __init__(self, root, train_set_type='gt', test_set_type='det'):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, self.img_dir)
        self.train_pkl = osp.join(self.det_result_dir, 'prw_train.pkl')
        self.gallery_pkl = osp.join(self.det_result_dir, 'prw_test.pkl')
        self.query_txt = osp.join(self.dataset_dir, 'query_info.txt')
        self.query_pid_to_img = {}
        self.imgid_to_name = self.convert_imgid2name(self.img_dir)

        self._check_before_run()
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_pkl, train_set_type)
        query, num_query_pids, num_query_imgs = self._process_query(self.query_txt)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_pkl, test_set_type, use_unlabeled=True)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> prw loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")
        print("test_type: ",test_set_type)
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def convert_imgid2name(self, imgdir):
        cnt=1
        allimgs = glob.glob(osp.join(imgdir, '*.jpg'))
        allimgs.sort()
        imgid2name={}
        for img in allimgs:
            imgname=img.split('/')[-1]
            imgid2name['s'+str(cnt)+'.jpg']=imgname
            cnt+=1
        return imgid2name

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_pkl):
            raise RuntimeError("'{}' is not available".format(self.train_pkl))
        if not osp.exists(self.gallery_pkl):
            raise RuntimeError("'{}' is not available".format(self.gallery_pkl))

    def _process_dir(self, pkl_path, type='gt', use_unlabeled=False):
        with open(pkl_path, 'rb') as f:
            det_results = pickle.load(f)
        
        dataset = []
        pids = set()
        # num_imgs represent the number of labeled instance
        num_imgs = 0
        for img_data in det_results:
            img_path = os.path.join(self.img_dir, self.imgid_to_name[img_data['im_name']])
            if type == 'det' or type == 'mixed':
                for i in range(img_data['det_box_num']):
                    box = img_data['det_boxes'][i]
                    pid = img_data['det_pids'][i]
                    if pid == -2:
                        pid = -1
                    if pid ==-1 and not use_unlabeled:
                        continue
                    iou = img_data['det_ious'][i]
                    score = img_data['det_scores'][i]
                    
                    if self.query_pid_to_img != {} and pid != -1 and img_path in self.query_pid_to_img[pid]:
                        continue
                    dataset.append((img_path, box, pid, iou))
                    pids.add(pid)
                    num_imgs += 1

            if type != 'det':
                for i in range(img_data['gt_box_num']):
                    box = img_data['gt_boxes'][i]
                    pid = img_data['gt_pids'][i]

                    if pid == -1 or pid == -2:
                        continue
                    iou = 1.0

                    if self.query_pid_to_img != {} and img_path in self.query_pid_to_img[pid]:
                        continue
                    dataset.append((img_path, box, pid, iou))
                    pids.add(pid)
                    num_imgs += 1
        return dataset, len(pids), num_imgs

    def _process_query(self, query_txt):
        dataset = []
        # map testID to 0 ~ N-1
        testID=loadmat(osp.join(self.dataset_dir,'ID_test.mat'))
        testID=testID['ID_test2'][0].tolist()
        testID.sort()
        dict4testID=dict(zip(testID,[x for x in range(len(testID))]))

        test = open(query_txt,'r')
        test = test.readlines()
        for item in test:
            query =item.strip().split(' ')
            # query
            im_name = self.img_dir+'/'+query[-1]+'.jpg'
            box = np.asarray(query[1:5],dtype='float64')
            box[2] += box[0]
            box[3] += box[1]
            pid = dict4testID[int(query[0])]
            dataset.append((im_name, box, pid, 1.0))
            if pid in self.query_pid_to_img.keys():
                self.query_pid_to_img[pid].append(im_name)
            else:
                self.query_pid_to_img[pid]=[im_name]
        num_pids = len(self.query_pid_to_img.keys())
        num_imgs = len(test)
        #print(dataset[0])
        return dataset, num_pids, num_imgs
"""Create dataset"""

__img_factory = {
    'psdb': PSDB,
    'PRW':PRW
}

def get_names():
    return list(__img_factory.keys())

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)
