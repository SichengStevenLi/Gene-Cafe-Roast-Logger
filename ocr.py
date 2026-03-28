from __future__ import annotations

import cv2
import numpy as np


SEGMENT_MAP: dict[int, set[str]] = {
    0: set("ABCDEF"),
    1: set("BC"),
    2: set("ABDEG"),
    3: set("ABCDG"),
    4: set("BCFG"),
    5: set("ACDFG"),
    6: set("ACDEFG"),
    7: set("ABC"),
    8: set("ABCDEFG"),
    9: set("ABCDFG"),
}


def get_ocr_status() -> tuple[bool, str]:
    return True, "Strict seven-segment decoder ready"


def _crop_inner(img: np.ndarray, frac_x: float, frac_y: float) -> np.ndarray:
    h, w = img.shape[:2]
    mx = max(1, int(w * frac_x))
    my = max(1, int(h * frac_y))
    if (w - 2 * mx) < 20 or (h - 2 * my) < 20:
        return img
    return img[my : h - my, mx : w - mx]


def _extract_display_window(roi_bgr: np.ndarray) -> np.ndarray:
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr

    img = _crop_inner(roi_bgr, frac_x=0.02, frac_y=0.03)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Keep anything that is not plain white border.
    keep = (gray < 245).astype(np.uint8) * 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 0.35 * img.shape[0] * img.shape[1]:
        return img

    display = img[y : y + h, x : x + w]
    # Remove the white outer frame more aggressively.
    return _crop_inner(display, frac_x=0.03, frac_y=0.08)


def _digit_boxes(display_bgr: np.ndarray) -> list[tuple[int, int]]:
    """
    Fixed three-digit split. This is more stable for your display than trying to
    infer separators from glare.
    """
    h, w = display_bgr.shape[:2]
    boxes: list[tuple[int, int]] = []

    for i in range(3):
        cx = (i + 0.5) * w / 3.0
        half = 0.145 * w
        x0 = int(max(0, cx - half))
        x1 = int(min(w, cx + half))
        boxes.append((x0, x1))

    return boxes


def _hot_map(digit_bgr: np.ndarray) -> np.ndarray:
    """
    Build a 'lit segment' map.
    Lit segments are usually bright and relatively desaturated compared to the tan/off segments.
    """
    hsv = cv2.cvtColor(digit_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # White-hot LED tends to have high V and lower S than the beige/off segments.
    hot = np.clip(v - 0.72 * s, 0.0, 1.0)

    bg = float(np.percentile(hot, 30))
    hot = np.clip(hot - bg, 0.0, 1.0)

    mx = float(hot.max())
    if mx > 1e-6:
        hot = hot / mx

    # Slight gamma to suppress weak glow.
    hot = hot ** 1.15
    return hot


def _segment_scores(digit_bgr: np.ndarray) -> dict[str, float]:
    hot = _hot_map(digit_bgr)
    h, w = hot.shape[:2]

    # Narrower windows reduce bleed from neighboring segments.
    rects = {
        "A": (0.25, 0.06, 0.75, 0.15),
        "B": (0.69, 0.16, 0.84, 0.42),
        "C": (0.69, 0.53, 0.84, 0.79),
        "D": (0.25, 0.79, 0.75, 0.90),
        "E": (0.13, 0.53, 0.28, 0.79),
        "F": (0.13, 0.16, 0.28, 0.42),
        "G": (0.25, 0.43, 0.75, 0.57),
    }

    scores: dict[str, float] = {}
    for seg, (x0f, y0f, x1f, y1f) in rects.items():
        x0 = int(x0f * w)
        x1 = int(x1f * w)
        y0 = int(y0f * h)
        y1 = int(y1f * h)

        roi = hot[y0:y1, x0:x1]
        if roi.size == 0:
            scores[seg] = 0.0
            continue

        # 75th percentile works better here than mean because the segment glow is uneven.
        scores[seg] = float(np.percentile(roi, 75))

    return scores


def _digit_candidate_scores(seg: dict[str, float]) -> dict[int, float]:
    scores: dict[int, float] = {}
    on_threshold = 0.58

    for digit, expected_on in SEGMENT_MAP.items():
        expected_off = set("ABCDEFG") - expected_on

        on_scores = np.array([seg[s] for s in expected_on], dtype=np.float32)
        off_scores = np.array([seg[s] for s in expected_off], dtype=np.float32) if expected_off else np.array([0.0], dtype=np.float32)

        on_mean = float(on_scores.mean()) if on_scores.size else 0.0
        off_mean = float(off_scores.mean()) if off_scores.size else 0.0

        on_hits = float((on_scores > on_threshold).mean()) if on_scores.size else 0.0
        off_hits = float((off_scores > on_threshold).mean()) if off_scores.size else 0.0

        score = 1.40 * on_mean - 1.00 * off_mean + 0.35 * (on_hits - off_hits)

        # Extra digit-specific penalties to avoid false 8/0/1/7 reads.
        if digit == 8:
            # 8 should have all segments convincingly on.
            score -= 1.60 * max(0.0, 0.68 - float(on_scores.min()))
        elif digit == 0:
            # 0 should not have middle bar strong.
            score -= 1.20 * seg["G"]
        elif digit == 4:
            # 4 should not have strong top or bottom bars.
            score -= 0.70 * max(0.0, seg["A"] - 0.35)
            score -= 0.70 * max(0.0, seg["D"] - 0.35)
        elif digit == 1:
            # 1 should not strongly light many other segments.
            score -= 0.90 * (seg["A"] + seg["D"] + seg["E"] + seg["F"] + seg["G"]) / 5.0
        elif digit == 7:
            # 7 should not look like a fully formed lower half.
            score -= 0.80 * (seg["D"] + seg["E"] + seg["F"] + seg["G"]) / 4.0

        scores[digit] = float(score)

    return scores


def _decode_digit(digit_bgr: np.ndarray) -> tuple[int | None, float]:
    seg = _segment_scores(digit_bgr)
    cand = _digit_candidate_scores(seg)

    ranked = sorted(cand.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        return None, 0.0

    best_digit, best_score = ranked[0]
    second_digit, second_score = ranked[1]

    gap = float(best_score - second_score)

    # Reject very ambiguous digits instead of forcing a wrong answer.
    if gap < 0.12:
        return None, 0.0

    # Also reject if the best candidate itself is weak.
    if best_score < 0.22:
        return None, 0.0

    conf = float(np.clip(0.35 + 2.6 * gap, 0.0, 1.0))
    return int(best_digit), conf


def read_temperature_from_frame(roi_bgr: np.ndarray) -> tuple[int | None, float]:
    if roi_bgr is None or roi_bgr.size == 0:
        return None, 0.0

    display = _extract_display_window(roi_bgr)
    if display is None or display.size == 0:
        return None, 0.0

    boxes = _digit_boxes(display)
    if len(boxes) != 3:
        return None, 0.0

    digits: list[int] = []
    confs: list[float] = []

    for x0, x1 in boxes:
        if x1 - x0 < 15:
            return None, 0.0

        digit_img = display[:, x0:x1]
        digit, conf = _decode_digit(digit_img)
        if digit is None:
            return None, 0.0

        digits.append(digit)
        confs.append(conf)

    value = int(digits[0] * 100 + digits[1] * 10 + digits[2])

    # Hard sanity check for Gene Cafe temperatures.
    if value < 100 or value > 550:
        return None, 0.0

    confidence = float(np.mean(confs)) if confs else 0.0

    # Reject low-confidence whole-number reads too.
    if confidence < 0.45:
        return None, 0.0

    return value, confidence
