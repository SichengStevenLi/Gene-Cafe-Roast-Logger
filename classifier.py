from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ClassifierResult:
    view_mode: str
    current_temp: int | None
    set_temp: int | None
    confidence: float


class TempClassifier:
    """
    Classifies readings into SET_VIEW vs CURRENT_VIEW without a visual indicator.
    Uses a short equal-duration rule when the read matches the set temp.
    """

    def __init__(self, equal_duration_sec: float = 0.2, tolerance: int = 2):
        self.equal_duration_sec = equal_duration_sec
        self.tolerance = tolerance
        self.equal_since_t: float | None = None

    def force_initial(self, set_temp: int) -> ClassifierResult:
        self.equal_since_t = None
        return ClassifierResult(
            view_mode="SET_VIEW",
            current_temp=set_temp,
            set_temp=set_temp,
            confidence=1.0,
        )

    def update(self, set_temp: int | None, raw_read: int | None, confidence: float, t_sec: float) -> ClassifierResult:
        if set_temp is None:
            self.equal_since_t = None
            return ClassifierResult("UNKNOWN", None, set_temp, confidence)

        # Do not reset equal_since_t on OCR miss.
        if raw_read is None:
            return ClassifierResult("UNKNOWN", None, set_temp, confidence)

        is_equal = abs(raw_read - set_temp) <= self.tolerance

        if not is_equal:
            self.equal_since_t = None
            return ClassifierResult("CURRENT_VIEW", raw_read, set_temp, confidence)

        if self.equal_since_t is None:
            self.equal_since_t = t_sec

        equal_elapsed = max(0.0, float(t_sec) - float(self.equal_since_t))
        if equal_elapsed >= self.equal_duration_sec:
            return ClassifierResult("CURRENT_VIEW", set_temp, set_temp, confidence)

        return ClassifierResult("SET_VIEW", None, set_temp, confidence)
