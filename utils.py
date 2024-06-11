from torch import Tensor, uint8
import torch
import torch.nn as nn


def collate_fn(batch):
    return tuple(zip(*batch))


def to_absolute_bbox_format(bbox: list[float], img_width: int, img_height: int) -> None:
    bbox[0] *= img_width
    bbox[1] *= img_height
    bbox[2] *= img_width
    bbox[3] *= img_height


def train_one_epoch(model, optimizer, data_loader, epoch):
    bbox_loss_sum = 0
    classification_loss_sum = 0
    count = 0
    for images, targets in data_loader:
        # targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        optimizer.zero_grad()
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        count += 1
        bbox_loss_sum += loss_dict['bbox_regression']
        classification_loss_sum += loss_dict['classification']
    print(f"-------------- EPOCH {epoch} ---------------")
    print(f'average bbox regression loss: {bbox_loss_sum / count}')
    print(f'average classification loss: {classification_loss_sum / count}')
    return bbox_loss_sum / count, classification_loss_sum / count


def rescale_image(image: Tensor):
    test_image_rescaled = (255.0 * (image - image.min()) / (image.max() - image.min())).to(uint8)
    return test_image_rescaled


def add_to_metric_fn(metric_fn, predictions, targets):
    rows = len(targets[0]['boxes'])
    pred_rows = len(predictions[0]['boxes'])
    targets_extended = torch.column_stack((targets[0]['boxes'], torch.zeros((rows, 3))))
    predictions_extended = torch.column_stack((predictions[0]['boxes'], torch.zeros((pred_rows, 1))))
    pkk = predictions[0]['scores'].reshape(-1, 1)
    predictions_extended = torch.column_stack((predictions_extended, pkk))
    metric_fn.add(predictions_extended.numpy(), targets_extended.numpy())


def print_evaluation_metrics(metric_fn, iou_thresholds):
    for threshold in iou_thresholds:
        print(f"test dataset mAP{int(threshold * 100)}: {metric_fn.value(iou_thresholds=threshold)['mAP']}")


def freeze_backbone(backbone):
    for param in backbone.parameters():
        param.requires_grad = False

def unfreeze_backbone(backbone):
    for param in backbone.parameters():
        param.requires_grad = True


