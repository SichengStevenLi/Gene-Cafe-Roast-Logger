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
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # Small capture buffer helps reduce stale/laggy frames.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self.cap = cap
            return

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self._open_capture()
            if self.cap is None or not self.cap.isOpened():
                return None, None

        ok, frame = self.cap.read()
        if (not ok) or (frame is None):
            # Recover once by reopening the device.
            self._open_capture()
            if self.cap is None or not self.cap.isOpened():
                return None, None

            # Some cameras need a short warmup after reopen.
            for _ in range(2):
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    break
                time.sleep(0.03)

            if (not ok) or (frame is None):
                return None, None

        x, y, w, h = self.roi
        h_frame, w_frame = frame.shape[:2]

        # Clamp ROI safely
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        roi = frame[y:y+h, x:x+w].copy()
        return frame, roi

    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None