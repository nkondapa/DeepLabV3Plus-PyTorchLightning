import os

from tqdm import tqdm
from PIL import Image

from datasets.segmentation._abstract import SegmentationDataset
from datasets.segmentation.pascal import PascalDataset
import numpy as np


class CocoDatasetPascalClasses(SegmentationDataset):
    classes = PascalDataset.classes
    ignore_index = PascalDataset.ignore_index
    visualizer_kwargs = PascalDataset.visualizer_kwargs

    def __init__(self, include_classes=None, return_dummy_prob=False, **kwargs):
        SegmentationDataset.__init__(self, **kwargs)
        self.return_dummy_prob = return_dummy_prob
        self.dataset_dir = os.path.join(self.data_dir, f"coco_voc/{self.split}")

        self.num_images = len(os.listdir(f"{self.dataset_dir}/img"))
        if not os.path.isdir(self.dataset_dir):
            raise ValueError(f"Dataset not found. Make sure directory is right or run export_all_images() first.")

        self.images = [f"{self.dataset_dir}/img/img_{i}.png" for i in range(self.num_images)]
        self.targets = [f"{self.dataset_dir}/mask/mask_{i}.png" for i in range(self.num_images)]

        if include_classes is not None and include_classes != "all":
            self.class_indices = [self.classes.index(c) for c in include_classes]

            # NOT NEEDED FOR SOFT INCLUDE -- TODO discuss with markus
            # self.class_indices.append(self.ignore_index)
            # self.class_indices.append(0)
            # self.class_indices = sorted(self.class_indices)

            new_images = []
            new_masks = []
            for index in range(len(self)):
                # img = Image.open(self.images[index]).convert("RGB")
                target = Image.open(self.targets[index])
                # filter out classes
                # This does only the include classes
                # if list(np.unique(np.array(target))) == self.class_indices:
                #     new_images.append(self.images[index])
                #     new_masks.append(self.targets[index])
                # soft include classes
                if len(set(np.unique(np.array(target))).intersection(set(self.class_indices))) > 0:
                    new_images.append(self.images[index])
                    new_masks.append(self.targets[index])

            self.images = new_images
            self.targets = new_masks

        self.num_images = len(self.images)

    def __len__(self):
        return self.num_images

    def _load_items(self, index: int):

        # img = Image.open(f"{self.dataset_dir}/img/img_{index}.png").convert('RGB')
        # mask = Image.open(f"{self.dataset_dir}/mask/mask_{index}.png")
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.targets[index])
        if self.return_dummy_prob:
            mask = np.concatenate([np.asarray(mask)[None], -1 * np.ones((len(self.classes), *mask.size[::-1]))], axis=0)

        return img, mask

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    ds = CocoDatasetPascalClasses(split="val")
    ds.export_all_images()

    ds = CocoDatasetPascalClasses(split="train")
    ds.export_all_images()
