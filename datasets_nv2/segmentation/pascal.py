import os
from typing import List, Tuple, Any

import numpy as np
from PIL import Image
from torchvision.datasets import VOCSegmentation
import numpy as np
from datasets.segmentation._abstract import SegmentationDataset

classes = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
           'horse', 'motorcycle', 'person', 'potted plant', 'sheep',
           'sofa', 'train', 'television']

class PascalDataset(SegmentationDataset, VOCSegmentation):
    classes = classes
    ignore_index = 255
    visualizer_kwargs = dict(palette="voc", fill_val="white")

    def __init__(self, include_classes=None, return_dummy_prob=False, **kwargs):
        SegmentationDataset.__init__(self, **kwargs, using_probs=return_dummy_prob)

        download = not os.path.isdir(os.path.join(self.data_dir, "VOCdevkit/VOC2012"))
        self.return_dummy_prob = return_dummy_prob
        VOCSegmentation.__init__(self, root=self.data_dir,
                                 image_set=self.split,
                                 year='2012',
                                 download=download,
                                 )

        ### reload dataset and filter out classes by self.class_indices
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

        print(f"Loaded {self.__class__.__name__} dataset with {len(self)} samples.")

    def __len__(self):
        return VOCSegmentation.__len__(self)

    def _load_items(self, index: int):
        img, mask = VOCSegmentation.__getitem__(self, index)
        if self.return_dummy_prob:
            mask = np.concatenate([np.asarray(mask)[None], -1 * np.ones((len(self.classes), *mask.size[::-1]))], axis=0).astype(np.float32)

        return img, mask


if __name__ == '__main__':
    ds = PascalDataset()
    x1, y1 = ds._load_items(0)
    x2, y2 = ds[0]
    print()
