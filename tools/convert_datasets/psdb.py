import argparse
import glob
import os.path as osp
import mat4py
from scipy.io import loadmat
import cv2
import mmcv
import numpy as np
import pycocotools.mask as maskUtils


def convert_data(img_dir, gt_dir,out_dir):
    test = loadmat(osp.join(gt_dir, 'pool.mat'))
    test = test['pool'].squeeze()
    test = [str(a[0]) for a in test]
    all_imgs = loadmat(osp.join(gt_dir, 'Images.mat'))
    all_imgs = all_imgs['Img'].squeeze()
    all_imgs = [str(a[0][0]) for a in all_imgs]
    train = list(set(all_imgs) - set(test))
    imgList, testImgList, annoList, testAnnoList = [],[],[],[]
    mmcv.mkdir_or_exist(out_dir+'train/')
    mmcv.mkdir_or_exist(out_dir+'test/')
    for img_file in glob.glob(osp.join(img_dir, '*.jpg')):
        img = cv2.imread(img_file)
        _filename = img_file.split('/')[-1]
        if _filename in train:
            cv2.imwrite(osp.join(out_dir,'train/'+_filename),img)
            imgList.append(dict(
                filename=_filename, height=img.shape[0], width=img.shape[1], id=_filename.split('.')[0]))
        else:
            cv2.imwrite(osp.join(out_dir,'test/'+_filename),img)
            testImgList.append(dict(
                filename=_filename, height=img.shape[0], width=img.shape[1], id=_filename.split('.')[0]))
    instance = mat4py.loadmat(osp.join(gt_dir, 'Person.mat'))
    instance = instance['Person']
    cateList = [{'id': 0, 'name': 'person'}]
    apperList = instance['nAppear']
    sceneList = instance['scene']
    print(sceneList)
    assert(len(apperList) == len(sceneList))
    iid = 0
    for it in range(len(apperList)):
        for index in range(int(apperList[it])):
            iid += 1
            _bbox = sceneList[it]["idlocate"][index]
            _imagename = sceneList[it]["imname"][index]
            if _imagename in train:
                annoList.append(
                    dict(image_id=_imagename.split('.')[0],
                         bbox=_bbox,
                         category_id=0,
                         id=str(iid),
                         iscrowd=0,
                         area=_bbox[2]*_bbox[3]
                         ))
            else:
                testAnnoList.append(
                    dict(image_id=_imagename.split('.')[0],
                         bbox=_bbox,
                         category_id=0,
                         id=str(iid),
                         iscrowd=0,
                         area=_bbox[2]*_bbox[3]
                         ))
    res = dict(train=dict(images=imgList, annotations=annoList, categories=cateList),
                test=dict(images=testImgList, annotations=testAnnoList, categories=cateList))
    mmcv.mkdir_or_exist(out_dir+'annotations/')
    mmcv.dump(res['train'], out_dir+'annotations/converted_train.json')
    mmcv.dump(res['test'], out_dir+'annotations/converted_test.json')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert psdb annotations to COCO format')
    parser.add_argument('--psdb_path', help='psdb data path',
                        default='../../data/psdb/dataset/')
    parser.add_argument('--img-dir', default='Image/SSM/', type=str)
    parser.add_argument('--gt-dir', default='annotation/', type=str)
    parser.add_argument('-o', '--out-dir', help='output path',
                        default='../../data/converted_psdb/')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    psdb_path = args.psdb_path
    out_dir = args.out_dir
    mmcv.mkdir_or_exist(out_dir)

    img_dir = osp.join(psdb_path, args.img_dir)
    gt_dir = osp.join(psdb_path, args.gt_dir)
    convert_data(img_dir, gt_dir,out_dir)


if __name__ == '__main__':
    main()
