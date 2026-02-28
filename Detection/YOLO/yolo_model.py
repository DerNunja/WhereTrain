from ultralytics import YOLO
from typing import List, Tuple, Optional
import numpy as np
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(_THIS_DIR, "best.pt")

YOLO_CONF_THRESHOLD = 0.3
YOLO_DEVICE = "cuda"

"""
names:
  0: Zug
"""

_yolo_model: Optional[YOLO] = None


def get_yolo_model(model_path: str = YOLO_MODEL_PATH) -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(model_path)
        _yolo_model.to(YOLO_DEVICE)
        print(f"YOLO model loaded from: {model_path} (device: {YOLO_DEVICE})")
    return _yolo_model


def detect_trains_yolo(warped_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Einzelbild-Inferenz. Lieber detect_trains_yolo_batch() verwenden."""
    return detect_trains_yolo_batch([warped_bgr])[0]


def detect_trains_yolo_batch(
    images: List[np.ndarray],
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Inferenz auf mehreren Bildern gleichzeitig (ein GPU-Aufruf).
    Gibt pro Bild eine Liste von BBoxen zurück: [[(x1,y1,x2,y2), ...], ...]
    """
    if not images:
        return []

    model = get_yolo_model()
    results = model(images, conf=YOLO_CONF_THRESHOLD, device=YOLO_DEVICE, verbose=False)

    all_bboxes: List[List[Tuple[int, int, int, int]]] = []
    for result in results:
        bboxes: List[Tuple[int, int, int, int]] = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) != 0:  # nur Klasse 0 = Zug
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        all_bboxes.append(bboxes)

    return all_bboxes