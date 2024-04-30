import os
import cv2
import numpy as np
from pathlib import Path
import json

import albumentations as A
import cv2

dataset_path = Path("data")


def scale_down(image: np.ndarray, height: int = 500) -> np.ndarray:
    """Scale down an image to a given height, keeping the aspect ratio.
    Args:
        image (np.ndarray): The image to scale down.
        height (int): The height to scale down to.
    Returns:
        np.ndarray: The scaled down image.
    """
    scale = height / image.shape[0]
    width = int(image.shape[1] * scale)
    return cv2.resize(image, (width, height))


def adjust_coordinates(
    coords: list[tuple], original_height: int, scaled_height: int
) -> list[tuple]:
    """
    Adjust the coordinates for a scaled-down image.

    Args:
    coords (list of tuples): List of (x, y) coordinates on the original image.
    original_height (int): Original height of the image.
    scaled_height (int): New height of the image after scaling.

    Returns:
    list of tuples: Adjusted coordinates for the scaled-down image.
    """
    scale_factor = scaled_height / original_height
    adjusted_coords = [
        (int(x * scale_factor), int(y * scale_factor)) for x, y in coords
    ]
    return adjusted_coords


def create_masks(data: dict[str, dict]) -> dict[str, np.ndarray]:
    """Create masks for the images in the dataset.

    Args:
        data (dict): The dataset information.

    Returns:
        dict: A dictionary containing the masks for each image."""
    masks = {}
    types = []
    for file in data.values():
        colors = {
            "ad": (1, 1, 1),
            "image": (11, 11, 11),
            "intro": (41, 41, 41),
            "subtitle": (71, 71, 71),
            "text": (101, 101, 101),
            "title": (201, 201, 201),
        }
        if not file["regions"]:
            continue
        img_path = dataset_path / "original" / file["filename"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        # Scale the image down and save it somewhere
        img_scaled = scale_down(img)
        filename = Path(file["filename"]).stem
        cv2.imwrite(
            str(dataset_path / "scaled_down" / "images" / (filename + ".png")),
            img_scaled,
        )
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        for region in file["regions"]:
            if (
                not region
                or not region["region_attributes"]
                or "type" not in region["region_attributes"]
            ):
                continue
            region_type = region["region_attributes"]["type"]
            color = colors[region_type]
            if region["shape_attributes"]["name"] == "rect":
                top_left = (
                    region["shape_attributes"]["x"],
                    region["shape_attributes"]["y"],
                )
                bottom_right = (
                    top_left[0] + region["shape_attributes"]["width"],
                    top_left[1] + region["shape_attributes"]["height"],
                )
                top_left, bottom_right = adjust_coordinates(
                    [top_left, bottom_right], img.shape[0], img_scaled.shape[0]
                )
                cv2.rectangle(mask, top_left, bottom_right, color, -1)
                colors[region["region_attributes"]["type"]] = (
                    color[0] + 1,
                    color[1] + 1,
                    color[2] + 1,
                )
            elif region["shape_attributes"]["name"] == "polygon":
                points = (
                    np.array(region["shape_attributes"]["all_points_x"]),
                    np.array(region["shape_attributes"]["all_points_y"]),
                )
                points = np.transpose(np.array(points))
                points = adjust_coordinates(points, img.shape[0], img_scaled.shape[0])
                points = np.array([points], dtype=np.int32)
                cv2.fillPoly(mask, [points], color)
                colors[region["region_attributes"]["type"]] = (
                    color[0] + 1,
                    color[1] + 1,
                    color[2] + 1,
                )
            types.append(region["region_attributes"]["type"])
        masks[file["filename"]] = mask
    return masks


def save_masks(masks: dict[str, np.ndarray]):
    """Save the masks to the disk.

    Args:
        masks (dict): Dictionary containing the masks for each image."""
    masks_path = dataset_path / "scaled_down" / "masks"
    masks_path.mkdir(exist_ok=True)
    for filename, mask in masks.items():
        stem = Path(filename).stem
        filepath = masks_path / f"{stem}.png"
        cv2.imwrite(str(filepath), mask)


def data_augmentation():
    """Perform data augmentation on the dataset."""
    data_path = dataset_path / "scaled_down"
    imgs_path = data_path / "images"
    masks_path = data_path / "masks"
    imgs = os.listdir(imgs_path)
    masks = os.listdir(masks_path)

    for img_name, mask_name in zip(imgs, masks):
        img_scaled = cv2.imread(str(imgs_path / img_name))
        mask = cv2.imread(str(masks_path / mask_name), cv2.IMREAD_GRAYSCALE)

        aug = A.HorizontalFlip(p=1)
        augmented = aug(image=img_scaled, mask=mask)
        img_hflip = augmented["image"]
        mask_hflip = augmented["mask"]

        aug = A.VerticalFlip(p=1)
        augmented = aug(image=img_scaled, mask=mask)
        img_vflip = augmented["image"]
        mask_vflip = augmented["mask"]

        aug = A.VerticalFlip(p=1)
        augmented = aug(image=img_hflip, mask=mask_hflip)
        img_hvflip = augmented["image"]
        mask_hvflip = augmented["mask"]

        cv2.imwrite(str(imgs_path / f"{Path(img_name).stem}_hflip.png"), img_hflip)
        cv2.imwrite(str(masks_path / f"{Path(mask_name).stem}_hflip.png"), mask_hflip)
        cv2.imwrite(str(imgs_path / f"{Path(img_name).stem}_vflip.png"), img_vflip)
        cv2.imwrite(str(masks_path / f"{Path(mask_name).stem}_vflip.png"), mask_vflip)
        cv2.imwrite(str(imgs_path / f"{Path(img_name).stem}_hvflip.png"), img_hvflip)
        cv2.imwrite(str(masks_path / f"{Path(mask_name).stem}_hvflip.png"), mask_hvflip)


if __name__ == "__main__":
    with open(dataset_path / "via_annotations.json", "r") as f:
        data_info = json.load(f)

    masks = create_masks(data_info)
    save_masks(masks)
    data_augmentation()
