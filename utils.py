from torch import Tensor, uint8
import torch

def collate_fn(batch):
    return tuple(zip(*batch))


def to_absolute_bbox_format(bbox: list[float], img_width: int, img_height: int) -> None:
    bbox[0] *= img_width
    bbox[1] *= img_height
    bbox[2] *= img_width
    bbox[3] *= img_height


def train_one_epoch(model, optimizer, data_loader, epoch):
    print(f"-------------- EPOCH {epoch} ---------------")
    for images, targets in data_loader:
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        optimizer.zero_grad()
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        print(loss_dict)


def rescale_image(image: Tensor):
    test_image_rescaled = (255.0 * (image - image.min()) / (image.max() - image.min())).to(uint8)
    return test_image_rescaled

def add_to_metric_fn(metric_fn, predictions, targets):
    rows = len(targets[0]['boxes'])
    pred_rows = len(predictions[0]['boxes'])
    targets_extended = torch.column_stack((targets[0]['boxes'], torch.zeros((rows, 3))))
    predictions_extended = torch.column_stack((predictions[0]['boxes'], torch.zeros((pred_rows, 1))))
    pkk = predictions[0]['scores'].reshape(-1,1)
    predictions_extended = torch.column_stack((predictions_extended, pkk))
    metric_fn.add(predictions_extended.numpy(), targets_extended.numpy())