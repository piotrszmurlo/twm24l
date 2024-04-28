from torch import Tensor, uint8


def collate_fn(batch):
    return tuple(zip(*batch))


def to_absolute_bbox_format(bbox: list[float], img_width: int, img_height: int) -> None:
    bbox[0] *= img_width
    bbox[1] *= img_height
    bbox[2] *= img_width
    bbox[3] *= img_height


def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
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
