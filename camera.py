import cv2

class Camera:
    def __init__(self, index=0, roi=(0, 0, 100, 100)):
        self.index = index
        self.roi = roi

        self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Try to reduce auto-brightening / bloom.
        # Some webcams ignore some of these, but they are worth trying.
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        except Exception:
            pass

        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -8)
        except Exception:
            pass

        try:
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
        except Exception:
            pass

        try:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
        except Exception:
            pass

        # Warm up camera a bit
        for _ in range(8):
            self.cap.read()

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return None, None

        # Flush buffered frames so we work on the newest image.
        for _ in range(3):
            self.cap.grab()

        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None, None

        x, y, w, h = self.roi
        h_frame, w_frame = frame.shape[:2]

        x = max(0, min(int(x), w_frame - 1))
        y = max(0, min(int(y), h_frame - 1))
        w = max(1, min(int(w), w_frame - x))
        h = max(1, min(int(h), h_frame - y))

        roi = frame[y:y + h, x:x + w].copy()
        return frame, roi

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
