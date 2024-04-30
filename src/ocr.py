from tesserocr import PyTessBaseAPI, RIL, PSM
import cv2
import easyocr
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import os
import json
from tqdm import tqdm
from loguru import logger
from collections import defaultdict

from src.mytypes import ImageSection
from src.utils import file_sort_order


def preprocess_image(image: str | PILImage.Image | np.ndarray, image_type):
    """Preprocess the image for OCR. Takes an image path, PIL Image or numpy array as input.

    Args:
        image (str | Image.Image | np.ndarray): The image to preprocess.

    Returns:
        PIL.Image: The preprocessed image."""
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, PILImage.Image):
        image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    padding = 10
    gray = cv2.copyMakeBorder(
        gray, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    if image_type == "title":
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open, iterations=2)
    else:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)

    if image_type == "title":
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
    else:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gray = cv2.bitwise_not(gray)
    gray = PILImage.fromarray(gray)
    return gray


def ocr_image_easyocr(image_path: str):
    """Perform OCR on the image using EasyOCR.

    Args:
        image_path (str): The path to the image to perform OCR on.

    Returns:
        Tuple[List[str], List[float], List[Tuple[int]]]: The words, confidences and bounding boxes.
    """
    reader = easyocr.Reader(["nl", "en", "de"])
    result = reader.readtext(image_path, decoder="wordbeamsearch", beamWidth=5)
    words = [r[1] for r in result]
    confidences = [float(r[2]) for r in result]
    bounding_boxes = [[[int(box) for box in boxes] for boxes in r[0]] for r in result]
    if len(words) == 0:
        logger.warning(f"No text found in {image_path}")
        words = [f"<NO TEXT FOUND: {Path(image_path).name}>"]
        confidences = [0]
        bounding_boxes = [(0, 0, 0, 0)]
    return words, confidences, bounding_boxes


def ocr_image_tesseract(image_path: str, image_type: ImageSection):
    """Perform OCR on the image. Ideally, the image is preprocessed with preprocess_image() but it can just be a PIL image.

    Args:
        image (str): The path to the image to perform OCR on.
        type (ImageSection): The type of image section.

    Returns:
        Tuple[List[str], List[float], List[Tuple[int]]]: The words, confidences and bounding boxes.
    """
    image = preprocess_image(image_path, image_type)
    if type == ImageSection.TITLE:
        psm = PSM.SINGLE_BLOCK
    else:
        psm = PSM.SINGLE_COLUMN
    with PyTessBaseAPI(
        psm=psm,
        lang=os.getenv("OCR_LANGUAGES"),
        path=os.getenv("OCR_TESSDATA"),
    ) as api:
        api.SetVariable("debug_file", "tesserocr.debug1.log")
        api.SetImage(image)
        api.Recognize()
        words = []
        confidences = []
        bounding_boxes = []
        ri = api.GetIterator()
        level = RIL.WORD
        while ri:
            try:
                word: str = ri.GetUTF8Text(level)
                conf: float = ri.Confidence(level) / 100
                if word:
                    words.append(word)
                    confidences.append(conf)
                    x1, y1, x2, y2 = ri.BoundingBox(level)
                    bounding_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                words.append(f"<NO TEXT FOUND: {Path(image_path).name}>")
                confidences.append(0)
                bounding_boxes.append((0, 0, 0, 0))
                logger.warning(f"Error in file {image_path}: {e}")
                break
            if not ri.Next(level):
                break
    return words, confidences, bounding_boxes


def ocr_dir(path: Path | str, output_path: Path | str = None):
    """Process all images in a directory. The directory should have one directory per image, and each image directory should have the segmented images.
    The segmented images need to end with i_type.png, where i is the index in the sorted order and type is the type of image section.

    Args:
        path (Path | str): The path to the directory.
        output_path (Path | str, optional): The path to save the output JSON file. Defaults to None.

    Returns:
        Dict[str, Dict[str, List]]: A dictionary with the results.
    """
    if isinstance(path, str):
        path = Path(path)
    if output_path is not None:
        if isinstance(output_path, str):
            output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f'Storing results in "{output_path}"')

    result = defaultdict(dict)

    for dir in tqdm(
        list(path.iterdir()), desc="OCR: Processing directories", leave=True
    ):
        if not dir.is_dir():
            continue
        result[dir.stem] = defaultdict(list)

        sorted_files = sorted(dir.iterdir(), key=file_sort_order)

        for file in tqdm(
            sorted_files, desc=f"OCR: Segments in {dir.stem}", leave=False, position=1
        ):
            if file.suffix not in [".png", ".jpg", ".jpeg"]:
                continue
            segment_type = file.stem.split("_")[-1]
            segment_type = ImageSection[segment_type.upper()]
            if segment_type == ImageSection.TEXT:
                words, confidences, bounding_boxes = ocr_image_tesseract(
                    str(file), segment_type
                )
            elif segment_type == ImageSection.TITLE:
                words, confidences, bounding_boxes = ocr_image_easyocr(str(file))
            else:
                continue
            result[dir.stem]["type"].append(segment_type.value)
            result[dir.stem]["words"].append(words)
            result[dir.stem]["confidences"].append(confidences)
            result[dir.stem]["bounding_boxes"].append(bounding_boxes)

        if output_path is not None:
            out_file = output_path / f"{dir.stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result[dir.stem], f, ensure_ascii=False, indent=4)

    return result
