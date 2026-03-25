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
      