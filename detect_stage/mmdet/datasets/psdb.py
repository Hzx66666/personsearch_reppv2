import itertools
import logging
import os.path as osp
import tempfile
import json
import pickle
import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset

from sklearn.metrics import average_precision_score, precision_recall_curve

@DATASETS.register_module()
class PsdbDataset(CustomDataset):

    CLASSES = ('person')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            print(info)
            info['filename'] = info['filename']
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                #gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, _bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]
    def xywh2xyxy(self, _bbox):

        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] + _bbox[0],
            _bbox[3] + _bbox[1],
        ]
    def _proposal2json(self, results):
        """Convert proposal results to COCO json style"""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style"""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = bboxes[i].tolist()
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style"""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def _compute_iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter * 1.0 / union

    def evaluate_detections(self,gt_roidb, gallery_det, det_thresh=0.5, iou_thresh=0.5):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        det_thresh (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """ 
        assert len(self)== len(gallery_det)

        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        
        for gt, det in zip(gt_roidb, gallery_det):
            det = np.asarray(det)
            gt_boxes=np.asarray(gt)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            #print(det)
                
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = self._compute_iou(gt_boxes[i,:4], det[j, :4])
            tfmat = (ious >= iou_thresh)

            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in range(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in range(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in range(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in range(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate

        print('  ap = ',ap)
        print('  recall = {:.2%}'.format(det_rate))
    
    def gen_reidpkl(self,gt_boxes, det_boxes, imganns, det_thresh=0.5):
        result = []
        gpids=[]
        for key in gt_boxes.keys():
            gt = np.asarray(gt_boxes[key])
            inds = np.where(gt[:, 4].ravel() !=-1)[0]
            gpids+=gt[inds].tolist()
            if key not in det_boxes.keys():
                result.append(dict(im_name='s'+str(key)+'.jpg',
                                gt_box_num=gt.shape[0],
                                det_box_num=0,
                                gt_boxes=gt[:, :4],
                                det_boxes=[],
                                det_scores=[],
                                gt_pids=gt[:, 4],
                                det_ious=[],
                                det_pids=[],
                                im_size=(
                                    imganns['s'+str(key)]['height'], imganns['s'+str(key)]['width'])
                                ))
                continue
            det = np.asarray(det_boxes[key])
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_gt = gt.shape[0]
            num_det = det.shape[0]
            det_pids = -np.ones(num_det,dtype=np.int32)
            det_ious = np.zeros(num_det)
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = self._compute_iou(gt[i, :4], det[j, :4])
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                if ious[largest_ind][j] != 0:
                    det_ious[j] = ious[largest_ind][j]
                    det_pids[j] = gt[largest_ind][4]
            result.append(dict(im_name='s'+str(key)+'.jpg',
                            gt_box_num=num_gt,
                            det_box_num=num_det,
                            gt_boxes=gt[:, :4],
                            det_boxes=det[:, :4],
                            det_scores=det[:, 4],
                            gt_pids=gt[:, 4],
                            det_ious=det_ious,
                            det_pids=det_pids,
                            im_size=(
                                    imganns['s'+str(key)]['height'], imganns['s'+str(key)]['width'])
                            ))
        with open('./result_pkl/test.pkl', 'wb') as pkf:
            pickle.dump(result, pkf)

    
    def evaluate(self,
                 results,
                 metric='bbox',
                 jsonfile_prefix=None,
                 ):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = {}
        cocoGt = self.coco
        f=open('./cocoGt.json','w')
        json.dump(cocoGt.anns,f)
        cocoDt=cocoGt.loadRes(result_files['bbox'])
        f=open('./cocoDt.json','w')
        json.dump(cocoDt.anns,f)
        gt_boxes={}
        det_boxes={}
        for anns in cocoGt.anns.values():
            ids= int(anns['image_id'][1:])
            if ids not in gt_boxes.keys():
                gt_boxes[ids]=[anns['bbox']+[anns['pid']]]
            else:
                gt_boxes[ids].append(anns['bbox']+[anns['pid']])
        for anns in cocoDt.anns.values():
            ids= int(anns['image_id'][1:])
            if ids not in det_boxes.keys():
                det_boxes[ids]=[anns['bbox']]
            else:
                det_boxes[ids].append(anns['bbox'])
        for key in gt_boxes.keys():
            for x in range(len(gt_boxes[key])):
                pid = int(gt_boxes[key][x][4])
                gt_boxes[key][x] = self.xywh2xyxy(
                    gt_boxes[key][x][:4])
                gt_boxes[key][x].append(pid)
        self.gen_reidpkl(gt_boxes,det_boxes,cocoGt.imgs)
        gt_boxes=list(gt_boxes.items())
        det_boxes=list(det_boxes.items())
        gt_boxes.sort(key=lambda x:x[0],reverse=False)
        det_boxes.sort(key=lambda x:x[0],reverse=False)
        for x in range(len(self)-len(det_boxes)):
            det_boxes.append((-1,[[0,0,0,0,0]]))
        for x in range(len(self)):
            if gt_boxes[x][0]!=det_boxes[x][0]:
                det_boxes.insert(x,(gt_boxes[x][0],[[0,0,0,0,0]]))
                det_boxes.pop()
        gt_boxes=[x[1] for x in gt_boxes]
        det_boxes=[x[1] for x in det_boxes]
        self.evaluate_detections(gt_boxes,det_boxes)
        return eval_results
    def evaluate2(self,
                 results,
                 metric='bbox',
                 jsonfile_prefix=None,
                 ):
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = {}
        cocoGt = self.coco
        f=open('./cocoGt.json','w')
        json.dump(cocoGt.anns,f)
        cocoDt=cocoGt.loadRes(result_files['bbox'])
        f=open('./cocoDt.json','w')
        json.dump(cocoDt.anns,f)
        f=open('./image.json','w')
        json.dump(cocoGt.imgs,f)
        gt_boxes={}
        det_boxes={}
        for anns in cocoGt.anns.values():
            ids= int(anns['image_id'][1:])
            if ids not in gt_boxes.keys():
                gt_boxes[ids]=[anns['bbox']+[anns['pid']]]
            else:
                gt_boxes[ids].append(anns['bbox']+[anns['pid']])
        for anns in cocoDt.anns.values():
            ids= int(anns['image_id'][1:])
            if ids not in det_boxes.keys():
                det_boxes[ids]=[anns['bbox']]
            else:
                det_boxes[ids].append(anns['bbox'])
        for key in gt_boxes.keys():
            for x in range(len(gt_boxes[key])):
                pid = int(gt_boxes[key][x][4])
                gt_boxes[key][x] = self.xywh2xyxy(
                    gt_boxes[key][x][:4])
                gt_boxes[key][x].append(pid)
        self.gen_reidpkl(gt_boxes,det_boxes,cocoGt.imgs)
        gt_boxes=list(gt_boxes.items())
        det_boxes=list(det_boxes.items())
        gt_boxes.sort(key=lambda x:x[0],reverse=False)
        det_boxes.sort(key=lambda x:x[0],reverse=False)
        for x in range(len(self)-len(det_boxes)):
            det_boxes.append((-1,[[0,0,0,0,0]]))
        for x in range(len(self)):
            if gt_boxes[x][0]!=det_boxes[x][0]:
                det_boxes.insert(x,(gt_boxes[x][0],[[0,0,0,0,0]]))
                det_boxes.pop()
        gt_boxes=[x[1] for x in gt_boxes]
        det_boxes=[x[1] for x in det_boxes]
        self.evaluate_detections(gt_boxes,det_boxes)
        return eval_results
    
    