# [AI tada] A Pipeline for Digitizing Dutch TV Guides Using Mask R-CNN and OCR

This repository contains the code for the AI tada TV guide part of the project. The goal of this project is to digitize Dutch TV guides using article segmentation models and OCR. 

NOTE: This code was only tested on Debian 12 under WSL 2 and not in Windows.

## Requirements
- Python 3.10+
- NLD tessdata (and optionally other languages) from [tessdata](https://github.com/tesseract-ocr/tessdata)

## Installation
1. Clone the repository and navigate to the directory
```bash
git clone git@github.com:FunnyPocketBook/ai-tada-tv-guide.git
cd ai-tada-tv-guide
```
2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Download the NLD tessdata from the [tessdata](https://github.com/tesseract-ocr/tessdata/blob/main/nld.traineddata) repository and place it in a folder, e.g. `tessdata`. In the `.env` file, adjust the path to the tessdata folder for `OCR_TESSDATA`.
5. Download the NCRV de Gids TV guide dataset and place the images in `data/images`. If you want to change the root data directory, make sure to keep the `images` subdirectory. Additionally, if the root data directory is changed, reflect the changes in the `.env` file for `DATA_INPUT_ROOT`.
    - If you want to train a model, include the [1980 TV Guide](https://archive.org/details/tv-guide-collection_202108/TV_Guide_Aug-09-1980_Small/) and [1988 TV Guide](https://archive.org/details/tv-guide-collection_202108/TV_Guide_Jul-05-11-1988_sm/) as well.

## Usage
The [main.py](main.py) file gives an example on how to run the pipeline. The pipeline consists of the following steps:
1. Article Segmentation: Segments the images found in `$DATA_INPUT_ROOT/images` and stores the segmented articles/images in `$DATA_OUTPUT_ROOT/images/boxes`.
2. OCR: Performs OCR on the segmented articles/images and stores the OCR results in `$DATA_OUTPUT_ROOT/text/unprocessed`.
3. Post-processing: Post-processes the OCR results and stores the post-processed results in `$DATA_OUTPUT_ROOT/text/processed`.


## Differences to the Report
The following changes were made to the pipeline described in the [report](report.pdf):
- The OCR step is now using [EasyOCR](https://github.com/JaidedAI/EasyOCR) in addition to Tesseract. EasyOCR is being used to detect the titles of the articles, while Tesseract is used for the rest of the text.
- The algorithm to find the ordering of the bounding boxes and their columns has been optimized. It no longer uses the Maximum Common Intersection (MCI) and instead uses vectorization to find the overlaps. Additionally, the bounding boxes are now sorted by their middle y-coordinate instead of their top y-coordinate.
- The RegEx patterns for the post-processing step have been slightly adjusted to better match the OCR results.
- This repository also contains labels for YOLOv8 as segmentation model.