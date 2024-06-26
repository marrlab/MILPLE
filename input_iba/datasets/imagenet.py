import os.path as osp

import cv2
import json
import numpy as np
from albumentations.core.composition import BboxParams
from functools import partial
from glob import glob

from .base import BaseDataset
from .builder import DATASETS, build_pipeline
from .utils import load_voc_bboxes
import torch


@DATASETS.register_module()
class ImageNet(BaseDataset):
    """ImageNet. The folders should be structured as follows:
        img_root/
            class_1/xxx.JPEG
            class_1/yyy.JPEG
            ...
            class_n/zzz.JPEG
            ...

        annot_root/
            class_1/xxx.xml
            class_1/xxx.xml
            ...
            class_n/xxx.xml

    Args:
        img_root (str): root of the images.
        ind_to_cls_file(str): json file that contains key-value pairs that
            map indices to class names or sub-folder names.
        pipeline (list): pipeline to transform the images.
        annot_root (str): root of the bounding box annotations
        with_bbox (bool): if True, load the bounding boxes.
    """

    def __init__(self,
                 img_root,
                 ind_to_cls_file,
                 pipeline,
                 annot_root=None,
                 with_bbox=False):
        super(ImageNet, self).__init__()
        if with_bbox:
            assert annot_root is not None
        self.img_root = img_root
        
        ######3
        #self.annot_root = annot_root
        
        
        self.with_bbox = with_bbox

        with open(ind_to_cls_file, 'r') as f:
            ind_to_cls = json.load(f)
        self.dir_to_ind = {v[0]: int(k) for k, v in ind_to_cls.items()}
        self.ind_to_dir = {v: k for k, v in self.dir_to_ind.items()}
        self.ind_to_cls = {int(k): v[1] for k, v in ind_to_cls.items()}
        self.cls_to_ind = {v: k for k, v in self.ind_to_cls.items()}

        # use albumentations.Compose
        self.image_paths = glob(
            osp.join(self.img_root, '**/*.npy'), recursive=True)
            
            #osp.join(self.img_root, '**/*.JPEG'), recursive=True)
            #osp.join(self.img_root, '**/*.tiff'), recursive=True)

        if self.with_bbox:
            annot_files = glob(
                osp.join(self.annot_root, '**/*.xml'), recursive=True)
            annot_file_names = list(
                map(lambda x: osp.splitext(osp.basename(x))[0], annot_files))
            self.image_paths = list(
                filter(
                    partial(_filter_fn, annot_file_names=annot_file_names),
                    self.image_paths))
            self.pipeline = build_pipeline(
                pipeline,
                default_args=dict(
                    bbox_params=BboxParams(
                        format='pascal_voc', label_fields=['labels'])))
        else:
            self.pipeline = build_pipeline(pipeline)

    def __getitem__(self, index):
        """Get a single sample.

        Args:
            index (int): index of sample.

        Returns:
            A tuple of:
                img (Tensor): img tensor with shape (3, H, W).
                target (int): class index.
                img_name (int): base name of the img file.
        """
        img_path = self.image_paths[index]
        #img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_folder, img_name_with_ext = osp.split(img_path)
        img_name = osp.splitext(img_name_with_ext)[0]
        synset = osp.basename(img_folder)
        target = int(self.dir_to_ind[synset])

        if self.with_bbox:
            annot_file = osp.join(self.annot_root, img_name + '.xml')
            annot = load_voc_bboxes(
                annot_file,
                name_to_ind_dict=self.dir_to_ind,
                ignore_difficult=False)
            bboxes = annot['bboxes']
            labels = annot['labels']
            # print(f'xml: {annot_file}, bboxes: {bboxes}')
            # albumentations
            #res = self.pipeline(image=img, bboxes=bboxes, labels=labels)
            # only keep the bboxes of a specific class
            bboxes = np.asarray(res['bboxes']).astype(int)
            labels = np.asarray(res['labels']).astype(int)
            res['bboxes'] = bboxes[labels == target]
        else:
            img = np.load(img_path)
            img = torch.tensor(img).permute(0, 3, 1, 2)
            #res = self.pipeline(image=img)
        #img = res['image']

        if self.with_bbox:
            bboxes = res['bboxes']
            return dict(
                input=img, target=target, input_name=img_name, bboxes=bboxes)
        else:
            return dict(input=img, target=target, input_name=img_name)

    def __len__(self):
        return len(self.image_paths)

    def get_ind_to_cls(self):
        """Get a dict mapping class indices to class names"""
        return self.ind_to_cls

    def get_cls_to_ind(self):
        """Get a dict mapping class names to class indices"""
        return self.cls_to_ind


def _filter_fn(img_path, annot_file_names):
    img_name = osp.splitext(osp.basename(img_path))[0]
    return img_name in annot_file_names
