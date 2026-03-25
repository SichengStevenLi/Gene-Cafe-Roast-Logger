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
        self.cap = cv2.VideoCapture(index)
        self.roi = roi

    def read(self):
        if not self.cap.isOpened():
            return None, None

        ok, frame = self.cap.read()
        if not ok:
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