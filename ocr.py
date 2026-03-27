from __future__ import annotations
import importlib
import cv2
import numpy as np


_PYTESSERACT = None


def _clean_binary(binary_img: np.ndarray) -> list[np.ndarray]:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_small)
    eroded = cv2.erode(closed, kernel_small, iterations=1)
    denoised = cv2.medianBlur(binary_img, 3)
    softened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel_med)

    return [binary_img, opened, closed, eroded, softened]


def _prepare_ocr_variants(roi_bgr) -> list[np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Upscaling and local contrast normalization help primitive seven-segment displays,
    # especially when bloom makes lit segments bleed into neighboring dark areas.
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    kernel_emphasis = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright_segments = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_emphasis)
    dark_segments = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel_emphasis)

    base_variants = [
        cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4
        ),
        cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
        ),
        cv2.threshold(bright_segments, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(dark_segments, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
    ]

    variants: list[np.ndarray] = []
    for img in base_variants:
        variants.extend(_clean_binary(img))
        variants.extend(_clean_binary(cv2.bitwise_not(img)))

    return variants

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

    variants = _prepare_ocr_variants(roi_bgr)

    # Find the best OCR result among variants
    best_value: int | None = None
    # Best confidence score found so far
  