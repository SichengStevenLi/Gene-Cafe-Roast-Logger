from __future__ import annotations
import importlib
import cv2
import numpy as np

_PYTESSERACT = None


def _get_pytesseract():
    global _PYTESSERACT
    if _PYTESSERACT is None:
        try:
            _PYTESSERACT = importlib.import_module("pytesseract")
        except Exception:
            _PYTESSERACT = False
    return _PYTESSERACT if _PYTESSERACT is not False else None


def get_ocr_status() -> tuple[bool, str]:
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


def _add_border(img: np.ndarray, border: int = 12, value: int = 255) -> np.ndarray:
    return cv2.copyMakeBorder(
        img,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=value,
    )


def _percentile_normalize(gray: np.ndarray) -> np.ndarray:
    p1 = np.percentile(gray, 1)
    p99 = np.percentile(gray, 99)
    if p99 <= p1:
        return gray
    scaled = (gray.astype(np.float32) - p1) * (255.0 / (p99 - p1))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _gamma_adjust(img: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(0.1, float(gamma))
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def _trim_roi(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return gray

    x_pad = max(1, int(w * 0.04))
    y_pad = max(1, int(h * 0.08))
    x2 = max(x_pad + 1, w - x_pad)
    y2 = max(y_pad + 1, h - y_pad)
    return gray[y_pad:y2, x_pad:x2]


def _prepare_ocr_variants(roi_bgr) -> list[np.ndarray]:
    if roi_bgr is None or roi_bgr.size == 0:
        return []

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = _trim_roi(gray)
    gray = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = _percentile_normalize(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Exposure rescue variants:
    # gamma > 1 darkens bright blown highlights
    # gamma < 1 brightens darker digits
    darker = _gamma_adjust(enhanced, 2.2)
    brighter = _gamma_adjust(enhanced, 0.8)

    kernel_emphasis = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bright_segments = cv2.morphologyEx(darker, cv2.MORPH_TOPHAT, kernel_emphasis)
    dark_segments = cv2.morphologyEx(brighter, cv2.MORPH_BLACKHAT, kernel_emphasis)

    base_images = [
        enhanced,
        darker,
        brighter,
        bright_segments,
        dark_segments,
    ]

    variants: list[np.ndarray] = []
    for base in base_images:
        bin_otsu = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        bin_otsu_inv = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        bin_adapt = cv2.adaptiveThreshold(
            base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4
        )
        bin_adapt_inv = cv2.adaptiveThreshold(
            base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
        )

        for img in [bin_otsu, bin_otsu_inv, bin_adapt, bin_adapt_inv]:
            img = cv2.medianBlur(img, 3)
            img = _add_border(img, 12, 255)
            variants.append(img)

    return variants


def read_temperature_from_frame(roi_bgr) -> tuple[int | None, float]:
    if roi_bgr is None:
        return None, 0.0

    ocr_ready, _ = get_ocr_status()
    if not ocr_ready:
        return None, 0.0

    pytesseract = _get_pytesseract()
    if pytesseract is None:
        return None, 0.0

    variants = _prepare_ocr_variants(roi_bgr)
    if not variants:
        return None, 0.0

    best_value: int | None = None
    best_conf = 0.0

    configs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
    ]

    for img in variants:
        for cfg in configs:
            try:
                data = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                    config=cfg,
                )
            except Exception:
                continue

            texts = data.get("text", [])
            confs = data.get("conf", [])

            for txt, conf_raw in zip(texts, confs):
                token = "".join(ch for ch in str(txt) if ch.isdigit())

                # Only accept exact 3-digit readings.
                if len(token) != 3:
                    continue

                try:
                    value = int(token)
                except ValueError:
                    continue

                # Plausible roast display range.
                if value < 100 or value > 550:
                    continue

                try:
                    conf = float(conf_raw)
                except (TypeError, ValueError):
                    continue

                if conf <= 0:
                    continue

                conf01 = max(0.0, min(1.0, conf / 100.0))

                if conf01 > best_conf:
                    best_conf = conf01
                    best_value = value

                    if conf01 >= 0.94:
                        return best_value, best_conf

    if best_value is None:
        return None, 0.0

    return best_value, best_conf
