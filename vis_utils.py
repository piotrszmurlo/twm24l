import os

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import Dataset, Subset, random_split
from torchvision.io import read_image
from torchvision.models import resnet18, VGG16_Weights, MobileNet_V3_Large_Weights
from torchvision.models.detection import ssd300_vgg16, ssdlite320_mobilenet_v3_large
from torchvision.ops import box_convert

from utils import to_absolute_bbox_format

DATASET_FOLDER = 'rescaled_data'
LOAD_MODEL = True
IMG_HEIGHT = 405
IMG_WIDTH = 720
NUM_EPOCHS = 100
UNFREEZE_BACKBONE_AFTER = 50
EVAL_MODEL_EVERY_X_EPOCHS = 10
SAVE_MODEL_EVERY_X_EPOCHS = NUM_EPOCHS // 4
MODEL_TO_LOAD = 'mobielnet.pth'
# MODEL_TO_LOAD = None
IOU_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# model = ssd300_vgg16(trainable_backbone_layers=0, weights = None, weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES, num_classes=2)
model = ssdlite320_mobilenet_v3_large(trainable_backbone_layers=0, weights = None, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT, num_classes=2)
model.load_state_dict(torch.load(MODEL_TO_LOAD))
model.eval()

class PlanesDataset(Dataset):
    def __init__(self, root, transforms, test=False):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, f"{DATASET_FOLDER}/images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, f"{DATASET_FOLDER}/labels"))))
        self.test = test

    def load_bboxes(self, idx: int) -> list[float]:
        label_path = os.path.join(self.root, f"{DATASET_FOLDER}/labels", self.labels[idx])
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                bbox = (list(map(float, line.split()[1:])))
                to_absolute_bbox_format(bbox, IMG_WIDTH, IMG_HEIGHT)
                bboxes.append(bbox)
        bboxes = box_convert(torch.as_tensor(bboxes), in_fmt='cxcywh', out_fmt='xyxy')
        return bboxes

    def load_image(self, idx: int) -> Tensor:
        image_path = os.path.join(self.root, f"{DATASET_FOLDER}/images", self.images[idx])
        image = read_image(image_path).float() / 255
        return image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        bboxes = self.load_bboxes(idx) # return list of [xmin, ymin, xmax, ymax]
        image = self.load_image(idx) # return an image tensor in shape [3, H, W]
        bbox_count = len(bboxes)

        labels = torch.ones((bbox_count,), dtype=torch.int64) # 1 - plane, 0 - background
        target = {
            "boxes": bboxes,
            "labels": labels,
        }

        # if self.transforms is not None:
        #   img, target = self.transforms(image, target)
        return image, target


    def __len__(self):
        return len(self.images)


dataset = PlanesDataset("", None)
indices = torch.arange(4)
dataset = Subset(dataset, indices)
indices = torch.randperm(len(dataset)).tolist()

# Hook setup
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def plot_activations(layer, num_cols=4, num_activations=16):
    num_kernels = layer.shape[1]
    fig, axes = plt.subplots(nrows=(num_activations + num_cols - 1) // num_cols, ncols=num_cols, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < num_kernels:
            ax.imshow(layer[0, i].cpu().numpy(), cmap='twilight')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

model.backbone.features[0].register_forward_hook(get_activation('bab'))
# model.head.classification_head.module_list[1].register_forward_hook(get_activation('baba'))
print(model)


with torch.no_grad():
    image, target = dataset[0]
    output = model([image])

# plot_activations(activations['bab'], num_cols=4, num_activations=16)
