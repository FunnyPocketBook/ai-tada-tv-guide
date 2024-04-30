from collections import defaultdict
import os
from pathlib import Path
import json
import re
from src.utils import levenshtein_distance


def load_ocr_results(ocr_result: str | Path | dict) -> dict:
    """Load OCR results from a file or use them directly if provided as a dictionary."""
    if isinstance(ocr_result, (str, Path)):
        with open(ocr_result, "r", encoding="utf-8") as f:
            return json.load(f)
    return ocr_result


def process_title(
    words: list[str],
    confidences: list[float],
    min_confidence: float,
    content: dict,
    last_broadcast_channel: str,
) -> str | None:
    """Process title data and update the last broadcast channel."""
    single_words = [word for item in words for word in item.split()]
    weekdays = [
        "maandag",
        "dinsdag",
        "woensdag",
        "donderdag",
        "vrijdag",
        "zaterdag",
        "zondag",
    ]
    # Skip the title if a weekday is included, since that is not a title but the header. Needs to be handled properly in the article segmentation part (annotate the headers separately)
    if any(
        levenshtein_distance(word, weekday) < 2
        for word in single_words
        for weekday in weekdays
    ):
        return last_broadcast_channel

    # If no programs were found for the last broadcast channel, remove it
    if (
        len(content[last_broadcast_channel]["programs"]) == 0
        and last_broadcast_channel in content
    ):
        del content[last_broadcast_channel]

    last_broadcast_channel = " ".join(
        word
        for word, confidence in zip(words, confidences)
        if confidence >= min_confidence
    )

    # If no words were found for the title during OCR or if the confidence is too low, skip the title
    if not last_broadcast_channel:
        return None

    content[last_broadcast_channel] = {"special": "", "programs": []}
    return last_broadcast_channel


def process_text(
    text: list[str],
    confidences: list[float],
    min_confidence: float,
    last_broadcast_channel: str,
    elk_uur_regex: str,
    program_regex: str,
    content: dict,
) -> None:
    """Process text data and extract programs."""
    text = " ".join(
        word
        for word, confidence in zip(text, confidences)
        if confidence >= min_confidence
    ).strip()

    # Check if the text contains the special text "Elk (heel) uur..."
    # If so, extract it and remove it from the text
    match = re.search(elk_uur_regex, text)
    if match:
        first_capturing_group = match.group(1)
        content[last_broadcast_channel]["special"] = first_capturing_group
        text = text.replace(first_capturing_group, "")

    matches = re.findall(program_regex, text)
    for m in matches:
        time, program = m
        # if the time is in the format of 0.00-01.30, split it and set an end time as well
        if "-" in time:
            times = time.split("-")
            program_info = {"time": times[0], "endtime": times[1], "program": program}
        else:
            program_info = {"time": time, "program": program}
        content[last_broadcast_channel]["programs"].append(program_info)


def text_postprocessing(ocr_result: str | Path | dict) -> dict:
    """Process the OCR results to extract the relevant information.

    Args:
        ocr_result (str | Path | dict): The OCR results as a file path or dictionary.

    Returns:
        dict: The processed OCR results."""
    ocr_result = load_ocr_results(ocr_result)
    min_confidence = float(os.getenv("OCR_CONFIDENCE"))
    results = {}
    time_regex = r"\b\d{1,2}\.\d{2}\b"
    program_regex = r"(\b\d{1,2}\.\d{2}(?:-\d{1,2}\.\d{2})?\b)\s(.*?)(?=\s\b\d{1,2}\.\d{2}(?:-\d{1,2}\.\d{2})?\b|$)"
    elk_uur_regex = r"^(?=.*Elk)(.*?)\.\s(\d+\.\d+.*?$)"

    for file, data in ocr_result.items():
        timestamp_sum = sum(
            1 for words in data["words"] for s in words if re.match(time_regex, s)
        )
        if timestamp_sum < int(os.getenv("MINIMUM_TIMESTAMPS")):
            continue

        content = defaultdict(lambda: {"special": "", "programs": []})
        last_broadcast_channel = f"<NO TITLE FOUND {file}>"

        for text_type, words, confidences in zip(
            data["type"], data["words"], data["confidences"]
        ):
            if text_type == "title":
                last_broadcast_channel = process_title(
                    words, confidences, min_confidence, content, last_broadcast_channel
                )
                if last_broadcast_channel is None:
                    continue
            elif text_type == "text":
                process_text(
                    words,
                    confidences,
                    min_confidence,
                    last_broadcast_channel,
                    elk_uur_regex,
                    program_regex,
                    content,
                )

        results[file] = content
    return results
