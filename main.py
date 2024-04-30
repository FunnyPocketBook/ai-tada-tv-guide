import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import json

from src.article_segmentation.article_segmentation import run_inference_on_directory
from src.article_segmentation.train_maskrcnn import get_model_instance_segmentation
from src.ocr import ocr_dir
from src.text_postprocessing import text_postprocessing


load_dotenv(override=True)

loguru_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

logger.add("main.log", format=loguru_format)

DATA_INPUT_ROOT = Path(os.getenv("DATA_INPUT_ROOT"))
DATA_OUTPUT_ROOT = Path(os.getenv("DATA_OUTPUT_ROOT"))

out_dir_unprocessed = DATA_OUTPUT_ROOT / "text" / "unprocessed"
out_dir_processed = DATA_OUTPUT_ROOT / "text" / "processed"


def start_article_segmentation():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model_instance_segmentation(7)
    model = model.to(device)
    model.load_state_dict(
        torch.load(os.getenv("SEGMENTATION_MODEL"), map_location=device)
    )
    logger.info("Model loaded.")

    run_inference_on_directory(
        model, DATA_INPUT_ROOT / "images", DATA_OUTPUT_ROOT, device
    )


def start_ocr():
    result = ocr_dir(DATA_OUTPUT_ROOT / "images" / "boxes", out_dir_unprocessed)
    with open(out_dir_unprocessed / "ocr_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f)
    for key in result:
        with open(out_dir_unprocessed / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(result[key], f)


def start_postprocessing():
    processed = text_postprocessing(out_dir_unprocessed / "ocr_result.json")

    for key in processed:
        with open(out_dir_processed / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(processed[key], f, indent=4)


if __name__ == "__main__":
    out_dir_unprocessed.mkdir(parents=True, exist_ok=True)
    out_dir_processed.mkdir(parents=True, exist_ok=True)

    start_article_segmentation()
    start_ocr()
    start_postprocessing()
