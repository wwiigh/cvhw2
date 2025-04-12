import os

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
import torch


class TrainDatasets(Dataset):
    def __init__(self, imgdir, jsondir, transform=None):
        self.imgdir = imgdir
        self.transform = transform
        self.coco = COCO(jsondir)
        self.image_id = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):

        image_id = self.image_id[idx]
        annotation_id = self.coco.getAnnIds(image_id)
        annotation = self.coco.loadAnns(annotation_id)

        img_path = os.path.join(self.imgdir, str(image_id) + ".png")
        image = Image.open(img_path)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        boxes = []
        labels = []

        for ann in annotation:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)
        image_id = torch.as_tensor(image_id)

        target = {
            "boxes": boxes,
            "labels": labels,
            "id": image_id
        }
        return image, target


class ValDatasets(Dataset):

    def __init__(self, imgdir, jsondir, transform=None):
        self.imgdir = imgdir
        self.transform = transform
        self.coco = COCO(jsondir)
        self.image_id = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, idx):

        image_id = self.image_id[idx]
        annotation_id = self.coco.getAnnIds(image_id)
        annotation = self.coco.loadAnns(annotation_id)

        img_path = os.path.join(self.imgdir, str(image_id) + ".png")
        image = Image.open(img_path)
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        boxes = []
        labels = []

        for ann in annotation:
            xmin, ymin, w, h = ann['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)
        image_id = torch.as_tensor(image_id)

        target = {
            "boxes": boxes,
            "labels": labels,
            "id": image_id
        }

        return image, target


class TestDatasets(Dataset):

    def __init__(self, imgdir, transform=None):
        self.data = []
        self.name = []
        self.transform = transform

        for img in os.listdir(imgdir):
            self.data.append(os.path.join(imgdir, img))
            self.name.append(int(img.split(".")[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        path = self.data[idx]
        img = Image.open(path)

        if self.transform:
            img = self.transform(img)

        return img, self.name[idx]


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_train_dataloader(imgdir, jsondir, transform=None,
                         batch_size=1, shuffle=False):
    """Get train dataloader"""
    train_dataset = TrainDatasets(imgdir, jsondir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=4,
                                  pin_memory=True, collate_fn=collate_fn)
    return train_dataloader


def get_val_dataloader(imgdir, jsondir, transform=None,
                       batch_size=1, shuffle=False):
    """Get val dataloader"""
    val_dataset = ValDatasets(imgdir, jsondir, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4,
                                collate_fn=collate_fn)
    return val_dataloader


def get_test_dataloader(imgdir, transform=None,
                        batch_size=1, shuffle=False):
    """Get test dataloader"""
    test_dataset = TestDatasets(imgdir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)
    return test_dataloader
