from abc import abstractmethod, ABC
from typing import Tuple, List, Union
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

import misc.joint_segmentation_transforms as JT

PIL_Image = Image.Image

class SegmentationDataset(ABC):
    """Abstract dataset class for segmentation datasets. Assumes all output images should be squared"""

    classes: List[str] = []
    ignore_index = 255
    visualizer_kwargs = {}

    def __init__(self,
                 split: str = "train",
                 data_dir: str = "./data/",
                 resize_mode: str = "center-crop",
                 img_size: int = 512,
                 to_tensor: bool = True,
                 using_probs: bool = False,
                 ):
        self.split = split
        self.data_dir = data_dir
        self.img_size = img_size
        self.to_tensor = to_tensor
        self.using_probs = using_probs

        self.resize_mode = resize_mode
        if resize_mode == "random-crop" and split != "train":
            self.resize_mode = "center-crop"  # no random crop for val for reproducibility


    def joint_pre_transform(self, img: PIL_Image, mask: Union[PIL_Image, torch.Tensor, np.ndarray]):
        if isinstance(mask, np.ndarray):
            mask = torch.FloatTensor(mask)
        if isinstance(mask, PIL_Image):
            assert img.size == mask.size
        else:
            assert img.size == mask.shape[-2:][::-1]

        smaller_dim = min(img.size)
        larger_dim = max(img.size)

        if self.resize_mode == "random-crop":
            transform = JT.RandomCrop(smaller_dim)
        elif self.resize_mode == "center-crop":
            transform = JT.CenterCrop(smaller_dim)
        elif self.resize_mode == "pad":
            transform = JT.CenterCrop(larger_dim)
        else:
            raise ValueError(f"Unknown resize mode: {self.resize_mode}")

        transform = JT.Compose([
            transform,
            JT.Resize(int(1.2 * self.img_size)),
            JT.CenterCrop(self.img_size),
        ])

        return transform(img, mask)

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @property
    def n_classes(self):
        return len(self.classes)

    @abstractmethod
    def _load_items(self, index: int) -> Tuple[PIL_Image, PIL_Image]:
        raise NotImplementedError

    def __getitem__(self, index: int):

        img, mask = self._load_items(index)

        img, mask = self.joint_pre_transform(img, mask)

        if self.to_tensor:
            img = T.ToTensor()(img)
            if self.using_probs:
                mask = T.Lambda(lambda x: torch.from_numpy(np.asarray(x, dtype='float32')))(mask)
            else:
                mask = T.Lambda(lambda x: torch.from_numpy(np.asarray(x, dtype='int64')))(mask)

        return img, mask



