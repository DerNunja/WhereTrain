from ultralytics import YOLO
from typing import List, Tuple, Optional
import numpy as np
import os

# Pfad relativ zu dieser Datei -> funktioniert unabhängig vom Arbeitsverzeichnis
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(_THIS_DIR, "best.pt")

YOLO_CONF_THRESHOLD = 0.1  # Confidence-Schwellwert (ggf. anpassen)

"""
names:
  0: Zug
"""

_yolo_model: Optional[YOLO] = None


def get_yolo_model(model_path: str = YOLO_MODEL_PATH) -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(model_path)
        print(f"YOLO model loaded from: {model_path}")
    return _yolo_model


def detect_trains_yolo(warped_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Führt YOLO-Inferenz auf dem gewarpten (normalisierten) Bild durch.
    Gibt eine Liste von BBoxen zurück: [(x1, y1, x2, y2), ...]
    Klasse 0 = Zug (entspricht 'names: 0: Zug').
    """
    model = get_yolo_model()
    results = model(warped_bgr, conf=YOLO_CONF_THRESHOLD, verbose=False)

    bboxes: List[Tuple[int, int, int, int]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0:  # nur Klasse 0 = Zug
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))

    return bboxes