"""
AUFGABE:
- Frame laden
- identisch skalieren wie beim Normalize-Tool
- per Homographie warpen
- YOLO-OBB (Polygon) einem Gleis zuordnen (Overlap mit band)

VORAUSSETZUNGEN:
- YOLO läuft auf dem *warped* Bild ODER
  OBB-Punkte werden vorher mit H gewarpt
"""

import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

Point = Tuple[int, int]



# CONFIG (ANPASSEN BITTE)

SECTION_NPY = "Sections/abschnitt_2__ids=TL9_TR10_BR8_BL5__1280x640__dict=DICT_ARUCO_ORIGINAL.npy"
TRACKMAP_JSON = "Sections/abschnitt_2__trackmap.json"

INPUT_IMAGE = "Pictures/Base.jpg"

# MUSS gleich sein wie im MultiNormalizeTool!
INPUT_SCALE = 0.5

MIN_OVERLAP_PX = 30


# =========================
# LOADERS
def load_homography(path: str) -> np.ndarray:
    H = np.load(path)
    assert H.shape == (3, 3)
    return H


def load_track_masks(trackmap_json: str) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
    """
    Baut pro Gleis eine Binärmaske aus dem 'band'-Polygon
    """
    data = json.load(open(trackmap_json, "r", encoding="utf-8"))

    w = data["canvas_size"]["w"]
    h = data["canvas_size"]["h"]
    shape_hw = (h, w)

    masks: Dict[str, np.ndarray] = {}

    for tr in data["tracks"]:
        tid = tr["track_id"]
        band = np.array(tr["band"], np.int32).reshape((-1, 1, 2))

        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [band], 255)
        masks[tid] = mask

    return shape_hw, masks


# =========================
# GEOMETRIE
def scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def warp_frame(frame: np.ndarray, H: np.ndarray, canvas_wh: Tuple[int, int]) -> np.ndarray:
    w, h = canvas_wh
    return cv2.warpPerspective(frame, H, (w, h))


def polygon_to_mask(poly: List[Point], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    arr = np.array(poly, np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [arr], 255)
    return mask


# =========================
# TRACK ASSIGNMENT
def assign_track(
    obb_poly_warped: List[Point],
    track_masks: Dict[str, np.ndarray],
    shape_hw: Tuple[int, int],
) -> Tuple[Optional[str], int]:

    det_mask = polygon_to_mask(obb_poly_warped, shape_hw)

    best_id = None
    best_overlap = 0

    for tid, tmask in track_masks.items():
        overlap = cv2.countNonZero(cv2.bitwise_and(det_mask, tmask))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = tid

    if best_overlap < MIN_OVERLAP_PX:
        return None, best_overlap

    return best_id, best_overlap


# =========================
# DEBUG DRAW
def draw_debug(canvas: np.ndarray, obb: List[Point], track_id: Optional[str], ov: int):
    vis = canvas.copy()
    poly = np.array(obb, np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [poly], True, (0, 0, 255), 2)

    txt = f"track={track_id} overlap={ov}px"
    cv2.putText(vis, txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return vis


# =========================
# MAIN (DEMO)
def main():
    # --- Load data ---
    H = load_homography(SECTION_NPY)
    shape_hw, track_masks = load_track_masks(TRACKMAP_JSON)
    h, w = shape_hw

    # --- Load frame ---
    frame = cv2.imread(INPUT_IMAGE)
    if frame is None:
        raise RuntimeError("Image not found")

    # --- IMPORTANT: same scale as Normalize-Tool ---
    frame = scale_frame(frame, INPUT_SCALE)

    # --- Warp to canvas ---
    canvas = warp_frame(frame, H, (w, h))

    # --- Example OBB (REPLACE with YOLO output in warped coords!) ---
    obb_warped = [(420, 280), (500, 260), (520, 320), (440, 340)]

    # --- Assign ---
    track_id, overlap = assign_track(obb_warped, track_masks, shape_hw)
    print("Assigned:", track_id, "Overlap:", overlap)

    # --- Debug ---
    vis = draw_debug(canvas, obb_warped, track_id, overlap)
    cv2.imshow("Track Assignment", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
