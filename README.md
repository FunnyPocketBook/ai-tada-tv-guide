# [AI tada] A Pipeline for Digitizing Dutch TV Guides Using Mask R-CNN and OCR

This repository contains the code for the AI tada TV guide part of the project. The goal of this project is to digitize Dutch TV guides using article segmentation models and OCR. 

NOTE: This code was only tested on Debian 12 under WSL 2 and not in Windows.

## Requirements
- Python 3.10+
- NLD tessdata (and optionally other languages) from [tessdata](https://github.com/tesseract-ocr/tessdata)

## Installation
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Download the NLD tessdata from the [tessdata](https://github.com/tesseract-ocr/tessdata/blob/main/nld.traineddata) repository and place it in a folder, e.g. `tessdata`. In the `.env` file, adjust the path to the tessdata folder for `OCR_TESSDATA`.
3. Download the NCRV de Gids TV guide dataset and place the images in `data/images`. If you want to change the root data directory, make sure to keep the `images` subdirectory. Additionally, if the root data directory is changed, reflect the changes in the `.env` file for `DATA_INPUT_ROOT`.
    - If you want to train a model, include the [1980 TV Guide](https://archive.org/details/tv-guide-collection_202108/TV_Guide_Aug-09-1980_Small/) and [1988 TV Guide](https://archive.org/details/tv-guide-collection_202108/TV_Guide_Jul-05-11-1988_sm/) as well.

## Usage
The [main.py](main.py) file gives an example on how to run the pipeline. The pipeline consists of the following steps:
1. Article Segmentation: Segments the images found in `$DATA_INPUT_ROOT/images` and stores the segmented articles/images in `$DATA_OUTPUT_ROOT/images/boxes`.
2. OCR: Performs OCR on the segmented articles/images and stores the OCR results in `$DATA_OUTPUT_ROOT/text/unprocessed`.
3. Post-processing: Post-processes the OCR results and stores the post-processed results in `$DATA_OUTPUT_ROOT/text/processed`.