import torch
from torch.utils.data import Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from stableDiff.create_class_embeddings import imagenet_templates


sentence_templates = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a picture of a {}.",
]

class PascalCategoriesDataset(Dataset):
    def __init__(self, split, cfg, transform=None, target_transform=None, batch_size=2, num_workers=0):
        self.split = split
        self.cfg = cfg
        self.transform = transform
        self.target_transform = target_transform

        # self._data = categories
        # self._data = [trainId2label[i].name for i in trainId2label]
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self._data = weights.meta["categories"]
        # skip background, TODO: double check
        self._data = self._data[1:]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_index = 255
        self.classes = self._data
        self.visualizer_kwargs = {}

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        item = self._data[index]
        # random template
        # template = imagenet_templates[torch.randint(0, len(imagenet_templates), (1,))]
        template = sentence_templates[torch.randint(0, len(sentence_templates), (1,))]
        res = template.format(item)
        return res

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers if not self.cfg["is_debug"] else 0,
                                           )


if __name__ == "__main__":
    dataset = PascalCategoriesDataset(split="train", cfg={"is_debug": True})
    print(dataset[0])
