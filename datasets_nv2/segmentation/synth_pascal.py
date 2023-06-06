import os
import re
from typing import List, Union
import numpy as np

from PIL import Image

from datasets.segmentation.pascal import PascalDataset
from datasets.segmentation._abstract import SegmentationDataset
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch


class SyntheticPascalDataset(SegmentationDataset):

    def __init__(self, directory: str = "./data/results",
                 include_classes: Union[List[str], str] = "all",
                 samples_per_class: int = None,
                 max_samples: int = None, # samples per class takes priority
                 return_prob: bool = False,

                 # filtering parameters
                 filter_method: str = None,
                 threshold: float = None,
                 topk: int = None,

                 *args, **kwargs
                 ):

        self.samples_per_class = samples_per_class
        self.return_prob = return_prob
        self.classes = PascalDataset.classes
        self.ignore_index = PascalDataset.ignore_index
        self.visualizer_kwargs = PascalDataset.visualizer_kwargs

        self.directory = directory

        super().__init__(*args, **kwargs, using_probs=return_prob)

        if include_classes == "all":
            self.include_classes = self.classes
        elif isinstance(include_classes, str):
            self.include_classes = [include_classes]
        else:
            self.include_classes = include_classes
        assert all([c in self.classes for c in self.include_classes]), f"One of classes unknown: {include_classes}"

        self.include_classes = [c for c in self.include_classes if c in self._available_classes]

        self.img_file_list, self.img_file_class = self._create_img_file_list()

        if filter_method is not None:
            cls_index = [i for i, c in enumerate(self.classes) if c in self.include_classes]
            cls_index.append(0)
            self.cls_index = sorted(cls_index)
            self.sample_filter = SampleFilter(self.img_file_list, self.cls_index, filter_method=filter_method)
            self.img_file_list = self.sample_filter.filter(topk=topk, threshold=threshold)
            # self.img_file_list = self._select_samples(threshold, use_cls_selection)

        if samples_per_class:
            max_samples = None
            self.img_file_class = np.array(self.img_file_class)
            indices = []
            for cls in self.include_classes:
                cls_indices = np.where(self.img_file_class == cls)[0]
                if len(cls_indices) > samples_per_class:
                    cls_indices = np.random.choice(cls_indices, size=samples_per_class, replace=False)
                indices.extend(cls_indices)

            new_img_file_list = []
            new_img_file_class = []
            for i in indices:
                new_img_file_class.append(self.img_file_class[i])
                new_img_file_list.append(self.img_file_list[i])
            self.img_file_list = new_img_file_list
            self.img_file_class = new_img_file_class

        if max_samples and max_samples < len(self.img_file_list):
            _indices = np.arange(len(self.img_file_list))
            _indices = np.random.choice(_indices, size=max_samples, replace=False)
            self.img_file_list = [self.img_file_list[i] for i in _indices]


    @property
    def _available_classes(self):
        """Return which classes are actually present in data folder."""
        return os.listdir(self.directory)

    def _create_img_file_list(self):
        files = []
        img_file_class = []
        for class_ in self.include_classes:
            cls_img_dir = os.path.join(self.directory, class_)
            img_files = [f for f in os.listdir(cls_img_dir) if re.match(r"\d+.png", f)]
            files.extend([os.path.join(self.directory, class_, img_file) for img_file in img_files])
            img_file_class.extend([class_] * len(img_files))
        return files, img_file_class

    def __len__(self):
        return len(self.img_file_list)

    def _load_items(self, index: int):

        img_filepath = self.img_file_list[index]
        img = Image.open(img_filepath).convert("RGB")

        mask_filepath = img_filepath[:-4] + "_mask.png"
        mask = Image.open(mask_filepath)

        if self.return_prob:
            prob_mask_filepath = img_filepath[:-4] + "_prob_mask.npy"
            prob_mask = np.load(prob_mask_filepath)
            mask = np.concatenate([np.asarray(mask)[None], prob_mask], axis=0)
        # mask_filepath = img_filepath[:-4] + "_mask.npy"

        # mask = np.load(mask_filepath)
        return img, mask

    def _select_samples(self, threshold, use_cls_selection):
        new_img_file_list = []

        for img_file in self.img_file_list:
            mask_filepath = img_file[:-4] + "_mask.npy"
            maskprob_filepath = img_file[:-4] + "_prob_mask.npy"
            mask = np.load(mask_filepath)
            maskprob = np.load(maskprob_filepath)
            # check sizes
            if mask.shape[0] != 512 or mask.shape[1] != 512:
                continue
            if use_cls_selection:
                if np.unique(mask).tolist() != self.cls_index:
                    continue
            if threshold:
                if np.mean(maskprob) < threshold:
                    continue
            new_img_file_list.append(img_file)

        return new_img_file_list


class SampleFilter:

    def __init__(self, img_file_list, cls_index, filter_method, **kwargs):
        self.img_file_list = img_file_list
        self.cls_index = cls_index
        self.kwargs = kwargs

        self.filter_method = filter_method
        method_dict = {
            'mean_filter': self.mean_filtering,
            'keep_only_target_classes': self.keep_only_target_class,
        }

        self.mask_target_class_means = []
        self.mask_target_class_object_ratio = []
        self.target_class_only = []
        self.filter_func = method_dict[filter_method]

    def filter(self, **kwargs):
        for img_file in self.img_file_list:
            mask_filepath = img_file[:-4] + "_mask.npy"
            maskprob_filepath = img_file[:-4] + "_prob_mask.npy"
            mask = np.load(mask_filepath)
            maskprob = np.load(maskprob_filepath)
            if mask.shape[0] != 512 or mask.shape[1] != 512:
                continue
            # plt.figure()
            # plt.hist(maskprob[self.cls_index[1:]][0].flatten())
            # plt.show()
            self.collect_filter_stats(mask_filepath, maskprob_filepath, mask, maskprob)

        self.filter_func(**kwargs)

        # debug -- visualize filtered image mean vs object size to image ratio
        # plt.figure()
        # plt.scatter(self.mask_target_class_means, self.mask_target_class_object_ratio)
        # plt.show()
        print(f'{len(self.img_file_list)} images after filtering')
        return self.img_file_list

    def collect_filter_stats(self, mask_filepath, maskprob_filepath, mask, maskprob):
        '''
        topk takes priority over threshold
        :param topk:
        :param threshold:
        :return: img_file_list
        '''

        if 'mean_filter' == self.filter_method:
            keep_mask = np.zeros_like(mask).astype(bool)

            # [1:] to ignore background class
            for cls_idx in self.cls_index[1:]:
                keep_mask = keep_mask | (mask == cls_idx)

            # evaluate mean value where target class is max across all classes to get the model confidence
            if keep_mask.sum() == 0:
                mv = -1
            else:
                mv = maskprob[self.cls_index[1:]][0, keep_mask].mean()
            # TODO remove: The below doesn't work because it is biased towards larger objects
            # mv = maskprob[self.cls_index[1:]].mean()

            # debug -- print out the model mean and the ratio of object pixels to total pixels
            # print('model_confidence', mv, np.sum(keep_mask.astype(int)) / (512 * 512))

            self.mask_target_class_means.append(mv)
            self.mask_target_class_object_ratio.append(np.sum(keep_mask.astype(int)) / (512 * 512))

        if 'keep_only_target_classes' == self.filter_method:
            if sorted(np.unique(mask).tolist()) != sorted(self.cls_index):
                self.target_class_only.append(False)
            else:
                self.target_class_only.append(True)

    def mean_filtering(self, topk=10, threshold=None, **kwargs):
        if topk is not None:
            self.mask_target_class_means = np.array(self.mask_target_class_means)
            indices = self.mask_target_class_means.argsort()[-topk:][::-1]
            self.img_file_list = [self.img_file_list[i] for i in indices]
        elif threshold is not None:
            self.img_file_list = [self.img_file_list[i] for i, v in enumerate(self.mask_target_class_means) if v > threshold]

    def keep_only_target_class(self, **kwargs):
        self.img_file_list = [self.img_file_list[i] for i, v in enumerate(self.target_class_only) if v]