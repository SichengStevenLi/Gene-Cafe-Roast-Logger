from __future__ import annotations
import importlib
import cv2
import numpy as np


_PYTESSERACT = None

# pytesseract reads Tesseract OCR results from images.
def _get_pytesseract():
    global _PYTESSERACT
    if _PYTESSERACT is None:
        try:
            _PYTESSERACT = importlib.import_module("pytesseract")
        except Exception:
            _PYTESSERACT = False
    return _PYTESSERACT if _PYTESSERACT is not False else None


def get_ocr_status() -> tuple[bool, str]:
    """
    Returns (is_ready, message) for OCR stack availability.
    """
    pytesseract = _get_pytesseract()
    if pytesseract is None:
        return (
            False,
            "OCR disabled: Python package 'pytesseract' is not installed.",
        )

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception:
        return (
            False,
            "OCR disabled: Tesseract binary is not installed or not on PATH. "
            "On macOS run: brew install tesseract",
        )

    return True, "OCR ready"

    """
    Returns (temp_int or None, confidence 0..1).

    Reads integer temperature from a ROI using Tesseract.

    Returns:
    - temp_int: parsed integer value or None when OCR is uncertain
    - confidence: 0..1 confidence score derived from Tesseract
    """
def read_temperature_from_frame(roi_bgr) -> tuple[int | None, float]:

    if roi_bgr is None:
        return None, 0.0

    # Ensure OCR stack is available
    ocr_ready, _ = get_ocr_status()
    if not ocr_ready:
        return None, 0.0

    pytesseract = _get_pytesseract()

    # Preprocess the image for better OCR results
    # Convert to grayscale
    # Apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold works better under changing lighting
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7
    )

    # Try a small set of variants to make OCR robust against display/background polarity.
    variants = [
        th,
        cv2.bitwise_not(th),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]

    # Find the best OCR result among variants
    best_value: int | None = None
    # Best confidence score found so far
    best_conf = 0.0
    # Tesseract OCR configuration
    ocr_config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"

    for img in variants:
        try:
            # data is a dict with keys: level, page_num, block_num, par_num, line_num,
            # word_num, left, top, width, height, conf, text
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config=ocr_config,
            )
        except Exception:
            continue

        # Parse Tesseract results
        # texts: list of recognized text strings
        # confs: list of confidence scores as strings
        texts = data.get("text", [])
        confs = data.get("conf", [])
        # Iterate over recognized texts and their confidence scores
        # pick the best valid integer with highest confidence
        for txt, conf_raw in zip(texts, confs):
            token = "".join(ch for ch in str(txt) if ch.isdigit())
            if not token:
                continue

            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                continue

            if conf <= 0:
                continue

            # Tesseract confidence is usually 0..100.
            conf01 = max(0.0, min(1.0, conf / 100.0))
            value = int(token)

            if conf01 > best_conf:
                best_conf = conf01
                best_value = value

    if best_value is None:
        return None, 0.0

    return best_value, best_conf