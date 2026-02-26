import cv2
import numpy as np
from typing import List, Tuple

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

def detect_by_color(
    bgr: np.ndarray,
    min_area: int = 550,           # filtert Kleinkram raus
    morph_kernel: int = 7,         # Morph-Kernelgröße
    morph_iters: int = 1,          # wie aggressiv säubern
) -> List[BBox]:
    """
    Findet grüne+gelbe Pixel (HSV), nimmt die größte zusammenhängende Region
    und gibt eine Bounding Box zurück (Liste mit 0 oder 1 BBox).

    Hinweis: HSV-Schwellen sind oft kamera/lighting-abhängig. Werte ggf. anpassen.
    """
    if bgr is None or bgr.size == 0:
        return []

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # --- HSV ranges (OpenCV Hue: 0..179) ---
    # Gelb: grob 20..35, Grün: grob 35..85 (kann je nach Beleuchtung variieren)
    yellow_lo = np.array([16, 130, 60], dtype=np.uint8)
    yellow_hi = np.array([34, 255, 255], dtype=np.uint8)

    green_lo  = np.array([50, 130, 40], dtype=np.uint8)
    green_hi  = np.array([85, 255, 255], dtype=np.uint8)

    mask_y = cv2.inRange(hsv, yellow_lo, yellow_hi)
    mask_g = cv2.inRange(hsv, green_lo, green_hi)
    mask = cv2.bitwise_or(mask_y, mask_g)

    # --- Morphology: Rauschen entfernen + Lücken schließen ---
    k = max(3, int(morph_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # erst "open" (entfernt kleine Punkte), dann "close" (schließt Löcher)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=max(1, morph_iters))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, morph_iters))

    # --- Connected Components: größte Region finden ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # stats: [label, x, y, w, h, area] (background label 0)
    best = None
    best_area = 0

    for lbl in range(1, num_labels):  # 0 = background
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, w, h)

    if best is None:
        return []

    x, y, w, h = best
    return [(int(x), int(y), int(x + w), int(y + h))]
