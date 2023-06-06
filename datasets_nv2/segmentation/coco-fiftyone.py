import os

from tqdm import tqdm
from PIL import Image, ImageFile
import fiftyone
import fiftyone.utils.labels as foul

from datasets.segmentation._abstract import SegmentationDataset
from datasets.segmentation.pascal import PascalDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CocoDatasetPascalClasses(SegmentationDataset):
    classes = PascalDataset.classes
    ignore_index = PascalDataset.ignore_index
    visualizer_kwargs = PascalDataset.visualizer_kwargs

    def __init__(self, **kwargs):
        SegmentationDataset.__init__(self, **kwargs)

        query_classes = [c for c in self.classes]
        query_classes[query_classes.index("background")] = "0"
        query_classes[query_classes.index("television")] = "tv"
        query_classes[query_classes.index("sofa")] = "couch"

        self.dataset = fiftyone.zoo.load_zoo_dataset(
        "coco-2017",
        split="validation" if self.split == "val" else "train",
        label_types=["segmentations"],
        classes=query_classes,
        max_samples=None,
        )

        foul.objects_to_segmentations(self.dataset, "ground_truth", "segmentations",
                                              mask_targets=dict(zip(list(range(len(query_classes))), query_classes))
                                              )

        self.sample_ids = self.dataset.values("id")

    def __len__(self):
        return len(self.sample_ids)

    def export_all_images(self):
        os.makedirs(f"data/coco_voc/{self.split}/img", exist_ok=True)
        os.makedirs(f"data/coco_voc/{self.split}/mask", exist_ok=True)

        for i in tqdm(range(len(self.sample_ids)), desc=f"Exporting {self.split} images and masks"):
            img, mask = self._load_items(i)
            img.save(f"data/coco_voc/{self.split}/img/img_{i}.png")
            mask.save(f"data/coco_voc/{self.split}/mask/mask_{i}.png")


    def _load_items(self, index: int):

        sample_id = self.sample_ids[index]
        sample = self.dataset.select(sample_id)
        filepath = sample.values("filepath")[0]
        img = Image.open(filepath)

        assert len(sample.values("segmentations")) == 1
        mask = sample.values("segmentations")[0].mask
        mask = Image.fromarray(mask)
        return img, mask




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    ds = CocoDatasetPascalClasses(split="train")
    ds.export_all_images()
    exit(0)

    ds = CocoDatasetPascalClasses(split="val")
    ds.export_all_images()
