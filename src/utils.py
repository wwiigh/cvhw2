import os

from torchvision import transforms
import torch.nn as nn
import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import random

# transform = A.Compose([
#             A.Rotate(limit=20, p=0.5),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2()
#         ], bbox_params=A.BboxParams(format='coco',,min_area=1, min_visibility=0.2))
def random_resize(img):
    size = random.choice([320, 480, 512, 640])
    return transforms.Resize((size, size))(img)

transform = transforms.Compose([
            # transforms.Lambda(lambda img: random_resize(img)),
            transforms.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 2.0)),
            # transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
            transforms.ToTensor(),
])


transform_val = transforms.Compose([
            transforms.ToTensor(),
])




