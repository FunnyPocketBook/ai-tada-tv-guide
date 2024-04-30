import os
import json
import torch
from torchvision.io import read_image, write_png
from PIL import Image
from src.article_segmentation.utils import get_transform
from pathlib import Path
from torchvision.utils import draw_bounding_boxes
from src.article_segmentation.train_maskrcnn import get_model_instance_segmentation
from tqdm import tqdm
from loguru import logger


def sort_bounding_boxes(boxes):
    """Sort bounding boxes by their y-coordinate and group them into columns based on intersection"""
    box_ids = torch.arange(len(boxes))

    normalized_boxes = boxes.clone()
    normalized_boxes[:, 1] = (
        0  # setting the top y to 0 for all boxes to create the intersections
    )

    x1 = normalized_boxes[:, 0].unsqueeze(1)
    x2 = normalized_boxes[:, 2].unsqueeze(1)

    min_x2 = torch.min(x2, x2.t())
    max_x1 = torch.max(x1, x1.t())

    intersections = min_x2 > max_x1

    # Track which column each box belongs to
    columns = [-1] * len(boxes)
    column_id = 0

    # Group boxes into columns based on intersections
    for i in range(len(boxes)):
        if columns[i] == -1:  # If not yet assigned to a column
            columns[i] = column_id
            # Assign intersecting boxes to the same column
            for j in range(len(boxes)):
                if intersections[i, j]:
                    columns[j] = column_id
            column_id += 1

    columns_dict = {}
    column_min_x = {}

    # find the minimum x-coordinate for each column to sort them
    for idx, col in enumerate(columns):
        box = boxes[idx].tolist()
        box_id = box_ids[idx].item()

        if col not in columns_dict:
            columns_dict[col] = [(box, box_id)]
            column_min_x[col] = box[0]
        else:
            columns_dict[col].append((box, box_id))
            column_min_x[col] = min(column_min_x[col], box[0])

    # Sort each column's boxes by the original y-value
    sorted_columns = []
    for col in sorted(
        column_min_x, key=column_min_x.get
    ):  # Sort columns by their minimum x-value
        sorted_boxes = sorted(
            columns_dict[col], key=lambda box: (box[0][1] + box[0][3]) / 2
        )  # Sort boxes within the column by y-value
        sorted_columns.append(sorted_boxes)

    sorted_columns = [box for column in sorted_columns for box in column]
    sorted_columns, sorted_indices = zip(*sorted_columns)

    return sorted_columns, sorted_indices


def run_inference_on_directory(
    model,
    input_dir,
    output_dir,
    device="cpu",
    threshold=0.85,
    save_image=False,
):
    """Run inference on all images in a directory and save the results to a JSON file."""
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    i2l = {1: "ad", 2: "image", 3: "intro", 4: "subtitle", 5: "text", 6: "title"}
    results = {}
    files: list[Path] = []

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) / "images"

    os.makedirs(output_dir, exist_ok=True)

    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            files.append(file)

    for file in tqdm(files, desc=f"Seg: Processing images: {file.stem}"):
        image = read_image(str(file))
        eval_transform = get_transform()

        model.eval()
        with torch.no_grad():
            x = eval_transform(image)
            x = x[:3, ...].to(device)
            predictions = model([x])
            pred = predictions[0]

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
            torch.uint8
        )
        image = image[:3, ...]

        high_confidence_indices = [
            i for i, score in enumerate(pred["scores"]) if score > threshold
        ]
        filtered_boxes = pred["boxes"][high_confidence_indices]
        filtered_scores = pred["scores"][high_confidence_indices].tolist()
        filtered_labels = [
            i2l[label.item()] for label in pred["labels"][high_confidence_indices]
        ]

        # Sort bounding boxes
        _, sorted_indices = sort_bounding_boxes(filtered_boxes.cpu())
        sorted_indices = torch.tensor(sorted_indices)
        filtered_boxes = filtered_boxes[sorted_indices]
        filtered_scores = [filtered_scores[i] for i in sorted_indices]
        filtered_labels = [filtered_labels[i] for i in sorted_indices]

        base_filename = file.stem
        boxes_dir = output_dir / "boxes" / (base_filename)
        os.makedirs(boxes_dir, exist_ok=True)

        # Save each bounding box as an image
        for i, (box, label) in enumerate(zip(filtered_boxes, filtered_labels)):
            x1, y1, x2, y2 = map(int, box)
            roi = image[:, y1:y2, x1:x2]
            if label == "intro":
                label = "text"
            elif label == "subtitle":
                label = "title"
            roi_path = str(boxes_dir / f"{base_filename}_{i}_{label}.png")
            write_png(roi, roi_path)

        # Draw bounding boxes on the image and save it if parameter is set
        if save_image:
            labels_with_scores = [
                f"{label}, {score:.3f}"
                for (label, score) in zip(filtered_labels, filtered_scores)
            ]
            image = draw_bounding_boxes(
                image,
                filtered_boxes,
                labels_with_scores,
                width=4,
                colors="lime",
                font="arial.ttf",
                font_size=50,
            )

            save_path = output_dir / f"{base_filename}.png"
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(save_path)

        boxes_list = filtered_boxes.tolist()
        results[file.stem] = [
            {"box": box, "label": label, "confidence": score}
            for (box, label, score) in zip(boxes_list, filtered_labels, filtered_scores)
        ]

    # Save results to JSON file
    json_path = os.path.join(output_dir, "inference_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f)

    logger.info(f"Inference complete. Results saved to {json_path}")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model_instance_segmentation(7)
    model = model.to(device)
    model.load_state_dict(
        torch.load(os.getenv("SEGMENTATION_MODEL"), map_location=device)
    )
    logger.info("Model loaded.")

    run_inference_on_directory(
        model,
        device,
    )
