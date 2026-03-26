import time
import platform
import cv2
from typing import Tuple

# takes image from cv2.VideoCapture and crops to ROI
# feeds the result to ocr.py to identify temperature digits
class Camera:
    """
    Simple camera reader with a fixed ROI crop.
    roi = (x, y, w, h) in pixels, relative to the full frame.
    """
    def __init__(self, index: int = 0, roi: Tuple[int, int, int, int] = (0, 0, 320, 180)):
        self.index = int(index)
        self.roi = roi
        self.cap = None
        self._open_capture()

    def _backend_candidates(self):
        if platform.system().lower() == "darwin":
            avf = getattr(cv2, "CAP_AVFOUNDATION", None)
            if avf is not None:
                return [avf, cv2.CAP_ANY]
        return [cv2.CAP_ANY]

    def _open_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        for backend in self._backend_candidates():
            cap = cv2.VideoCapture(self.in