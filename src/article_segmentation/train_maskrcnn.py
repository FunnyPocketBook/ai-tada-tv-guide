import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from src.article_segmentation.engine import train_one_epoch, evaluate
import src.article_segmentation.utils as utils


class TvGuideDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        boxes = masks_to_boxes(masks)

        labels = torch.zeros((num_objs,), dtype=torch.int64)
        # colors = {'ad': (1,1,1), 'image': (11,11,11), 'intro': (41,41,41), 'subtitle': (71,71,71), 'text': (101,101,101), 'title': (201,201,201)}
        for i, obj_id in enumerate(obj_ids):
            if obj_id < 11:
                labels[i] = 1
            elif obj_id < 41:
                labels[i] = 2
            elif obj_id < 71:
                labels[i] = 3
            elif obj_id < 101:
                labels[i] = 4
            elif obj_id < 201:
                labels[i] = 5
            else:
                labels[i] = 6

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = torch.zeros(
            (num_objs,), dtype=torch.int64
        )  # Not used for our purposes, since we have no crowds. However, it is required for the COCO evaluation.

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes,
    )
    return model


def train_model(
    dataset,
    dataset_val,
    num_classes,
    device,
    lr,
    step_size,
    gamma,
    optimizer_type="SGD",
    num_epochs=10,
    batch_size_train=2,
    batch_size_val=1,
):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    metric_loggers = []

    for epoch in range(num_epochs):
        metric_logger = train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq=30
        )
        metric_loggers.append(metric_logger)
        lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)

    print("Training complete.")
    return model


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = TvGuideDataset("dataset/train", utils.get_transform())
    dataset_val = TvGuideDataset("dataset/val", utils.get_transform())

    model = train_model(
        dataset,
        dataset_val,
        num_classes=5,
        device=device,
        lr=0.005,
        step_size=2,
        gamma=0.5,
        optimizer_type="SGD",
        num_epochs=5,
    )
    torch.save(model.state_dict(), "model.pth")
