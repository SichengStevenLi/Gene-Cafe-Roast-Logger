from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ClassifierResult:
    view_mode: str               # "CURRENT_VIEW" | "SET_VIEW" | "UNKNOWN"
    current_temp: int | None
    set_temp: int | None
    confidence: float


class TempClassifier:
    """
    Classifies readings into SET_VIEW vs CURRENT_VIEW without a visual indicator.
    Uses your idea plus a "consecutive equal" rule.
    """
    def __init__(self, equal_duration_sec: float = 5.0, tolerance: int = 0):
        # equal_duration_sec: how long temp must stay equal to set before we assume current == set
        # tolerance: allow +/- tolerance to count as equal
        self.equal_duration_sec = equal_duration_sec
        self.tolerance = tolerance
        self.equal_since_t: float | None = None

    def force_initial(self, set_temp: int) -> ClassifierResult:
        self.equal_since_t = None
        return ClassifierResult(
            view_mode="SET_VIEW",
            current_temp=set_temp,   # your x=0, y=set_temp anchor
            set_temp=set_temp,
            confidence=1.0,
        )

    def update(self, set_temp: int | None, raw_read: int | None, confidence: float, t_sec: float) -> ClassifierResult:
        if set_temp is None or raw_read is None:
            self.equal_since_t = None
            return ClassifierResult("UNKNOWN", None, set_temp, confidence)

        is_equal = abs(raw_read - set_temp) <= self.tolerance

        if not is_equal:
            self.equal_since_t = None
            # Different from set -> treat as current
            return ClassifierResult("CURRENT_VIEW", raw_read, set_temp, confidence)

        # Equal to set -> ambiguous
        if self.equal_since_t is None:
            self.equal_since_t = t_sec

        equal_elapsed = max(0.0, float(t_sec) - float(self.equal_since_t))
        if equal_elapsed >= self.equal_duration_sec:
            # If equal for long enough, assume current reached set
            return ClassifierResult("CURRENT_VIEW", set_temp, set_temp, confidence)

        # otherwise treat as set view (don’t log as current yet)
        return ClassifierResult("SET_VIEW", None, set_temp, confidence)