import torch.nn as nn
import torchvision.models as models
from dataset import get_train_dataloader
from utils import transform
from tqdm import tqdm
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



class Model(nn.Module):
    def __init__(self, training=True):
        super().__init__()
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        # self.model = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        # self.model = models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        self.training = training
        num_classes = 11
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, target=None):
        if target is not None:
            output = self.model(x, target)
        else:
            output = self.model(x)
        return output




def get_model(training=True):
    """Return Model with faster rcnn"""
    return Model(training=training)

if __name__ == "__main__":
    dataloader = get_train_dataloader("data/train","data/train.json",transform=transform)
    model = get_model(training=True)
    for image, target in tqdm(dataloader):
        print(type(target))
        x = model(image, target)
        print(x)
        break
