from ultralytics import YOLO
from typing import List, Tuple, Optional
import numpy as np

YOLO_MODEL_PATH = "best.pt"
YOLO_CONF_THRESHOLD = 0.3

model = YOLO(YOLO_MODEL_PATH)

"""
names:
  0: Zug
"""
  # Confidence-Schwellwert (ggf. anpassen)

_yolo_model: Optional[YOLO] = None

def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO model loaded from: {YOLO_MODEL_PATH}")
    return _yolo_model


def detect_trains_yolo(warped_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Führt YOLO-Inferenz auf dem gewarpten (normalisierten) Bild durch.
    Gibt eine Liste von BBoxen zurück: [(x1, y1, x2, y2), ...]
    Klasse 0 = Zug (entspricht 'names: 0: Zug' aus yolo_model.py).
    """
    model = get_yolo_model()
    results = model(warped_bgr, conf=YOLO_CONF_THRESHOLD, verbose=False)

    bboxes: List[Tuple[int, int, int, int]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0:   # nur Klasse 0 = Zug
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    return bboxes