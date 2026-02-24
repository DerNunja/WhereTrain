"""

- Homographie überspringen
- Bild nur auf feste Canvas bringen (Resize)
- Zug-OBB/Polygon per Overlap einem Gleis zuordnen
- Zusätzlich Position auf dem Gleis:
    s_norm     = Gleisanfang -> Gleisende (0..1)
    s_norm_rev = Gleisende  -> Gleisanfang (0..1)
    s_px       = Strecke entlang Mittellinie (Pixel)
    lateral_px = seitlicher Abstand zur Mittellinie (Pixel)

Abhängigkeiten:
- trackmap.json (aus map_tool.py)
"""

import cv2
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

Point = Tuple[int, int]
PtF = Tuple[float, float]

# =========================
# CONFIG

IMAGE_PATH = r"C:\Users\firas\Downloads\Kurve__ids=TL3_TR5_BR16_BL4__1280x640__dict=DICT_4X4_50 (2).png"
TRACKMAP_JSON = r"C:\Users\firas\Downloads\Kurve__ids=TL3_TR5_BR16_BL4__1280x640__dict=DICT_4X4_50__trackmap.json"

CANVAS_W, CANVAS_H = 1280, 640
MIN_OVERLAP_PX = 30

# Wenn dein Input-Bild nicht exakt 1280x640 ist, wird es auf diese Größe gezwungen.
FORCE_RESIZE_TO_CANVAS = True


# =========================
# DATA CLASSES

@dataclass
class Track:
    track_id: str
    polyline: List[Point]  # Mittellinie (Reihenfolge = Anfang->Ende)
    band: List[Point]      # Gleis-Bandpolygon (für Overlap)


@dataclass
class TrainTrackResult:
    train_idx: int
    track_id: Optional[str]
    overlap: int
    poly: List[Point]
    center_xy: PtF

    # Position
    s_px: Optional[float] = None        # Strecke entlang Mittellinie (Pixel)
    s_norm: Optional[float] = None      # Gleisanfang -> Gleisende (0..1)
    s_norm_rev: Optional[float] = None  # Gleisende -> Gleisanfang (0..1)
    lateral_px: Optional[float] = None  # Seitlicher Abstand zur Mittellinie (Pixel)


# =========================
# LOAD TRACKS + MASKS

def load_tracks_and_masks(trackmap_json: str):
    with open(trackmap_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    h, w = data["canvas_size"]["h"], data["canvas_size"]["w"]

    tracks: List[Track] = []
    masks: Dict[str, np.ndarray] = {}

    for tr in data["tracks"]:
        tid = tr["track_id"]
        polyline = [tuple(p) for p in tr["polyline"]]
        band = [tuple(p) for p in tr["band"]]

        tracks.append(Track(track_id=tid, polyline=polyline, band=band))

        band_arr = np.array(band, np.int32).reshape((-1, 1, 2))
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [band_arr], 255)
        masks[tid] = mask

    return (h, w), tracks, masks


# =========================
# OVERLAP ASSIGNMENT

def polygon_to_mask(poly: List[Point], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    arr = np.array(poly, np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [arr], 255)
    return mask


def assign_track(poly: List[Point], track_masks: Dict[str, np.ndarray], shape_hw: Tuple[int, int]):
    det_mask = polygon_to_mask(poly, shape_hw)

    best_id, best_ov = None, 0
    for tid, tmask in track_masks.items():
        ov = cv2.countNonZero(cv2.bitwise_and(det_mask, tmask))
        if ov > best_ov:
            best_id, best_ov = tid, ov

    if best_ov < MIN_OVERLAP_PX:
        return None, best_ov
    return best_id, best_ov


def polygon_center(poly: List[Point]) -> PtF:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (float(sum(xs)) / len(xs), float(sum(ys)) / len(ys))


# =========================
# POSITION ON POLYLINE
def project_point_to_segment(p: PtF, a: PtF, b: PtF):
    """
    Projektion eines Punktes p auf Segment a-b.
    Returns:
      t in [0,1], proj point q, squared distance
    """
    px, py = p
    ax, ay = a
    bx, by = b
    vx, vy = (bx - ax), (by - ay)
    wx, wy = (px - ax), (py - ay)

    vv = vx * vx + vy * vy
    if vv <= 1e-9:
        q = (ax, ay)
        dx, dy = (px - ax), (py - ay)
        return 0.0, q, dx * dx + dy * dy

    t = (wx * vx + wy * vy) / vv
    t = max(0.0, min(1.0, t))
    qx = ax + t * vx
    qy = ay + t * vy
    dx, dy = (px - qx), (py - qy)
    return t, (qx, qy), dx * dx + dy * dy


def polyline_lengths(polyline: List[Point]) -> Tuple[List[float], float]:
    seg_lens = []
    total = 0.0
    for i in range(len(polyline) - 1):
        x1, y1 = polyline[i]
        x2, y2 = polyline[i + 1]
        l = float(np.hypot(x2 - x1, y2 - y1))
        seg_lens.append(l)
        total += l
    return seg_lens, total


def position_on_track(center: PtF, track_polyline: List[Point]) -> Tuple[float, float, float]:
    """
    Gibt (s_px, s_norm, lateral_px) zurück.
    s_px: Distanz entlang polyline bis Projektion
    s_norm: s_px / total_len
    lateral_px: Abstand zur Mittellinie
    """
    if len(track_polyline) < 2:
        return 0.0, 0.0, 0.0

    seg_lens, total_len = polyline_lengths(track_polyline)
    if total_len <= 1e-9:
        return 0.0, 0.0, 0.0

    best_dist2 = float("inf")
    best_s = 0.0

    s_acc = 0.0
    p = center

    for i in range(len(track_polyline) - 1):
        a = (float(track_polyline[i][0]), float(track_polyline[i][1]))
        b = (float(track_polyline[i + 1][0]), float(track_polyline[i + 1][1]))

        t, _, dist2 = project_point_to_segment(p, a, b)
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_s = s_acc + t * seg_lens[i]

        s_acc += seg_lens[i]

    lateral = float(np.sqrt(best_dist2))
    s_norm = float(best_s / total_len)
    return best_s, s_norm, lateral


# =========================
# MAIN PIPELINE
def process_trains(
    obbs_canvas: List[List[Point]],
    tracks: List[Track],
    track_masks: Dict[str, np.ndarray],
    shape_hw: Tuple[int, int],
) -> List[TrainTrackResult]:
    results: List[TrainTrackResult] = []

    track_by_id = {t.track_id: t for t in tracks}

    for i, poly in enumerate(obbs_canvas):
        center = polygon_center(poly)
        tid, ov = assign_track(poly, track_masks, shape_hw)

        res = TrainTrackResult(
            train_idx=i,
            track_id=tid,
            overlap=ov,
            poly=poly,
            center_xy=center,
        )

        if tid is not None and tid in track_by_id:
            s_px, s_norm, lateral = position_on_track(center, track_by_id[tid].polyline)
            res.s_px = s_px
            res.s_norm = s_norm                 # Gleisanfang -> Gleisende
            res.s_norm_rev = 1.0 - s_norm        # Gleisende -> Gleisanfang
            res.lateral_px = lateral

        results.append(res)

    return results


# =========================
# DEBUG DRAW
def draw_debug(canvas: np.ndarray, tracks: List[Track], results: List[TrainTrackResult]) -> np.ndarray:
    out = canvas.copy()

    # Mittellinien der Gleise
    for t in tracks:
        pts = np.array(t.polyline, np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, (255, 255, 255), 1)
        # Track-ID am Anfang anzeigen
        x0, y0 = t.polyline[0]
        cv2.putText(out, f"{t.track_id} A", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x1, y1 = t.polyline[-1]
        cv2.putText(out, f"{t.track_id} E", (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Züge + Labels
    for r in results:
        poly = np.array(r.poly, np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [poly], True, (0, 0, 255), 2)

        cx, cy = int(r.center_xy[0]), int(r.center_xy[1])
        cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)

        if r.track_id is None:
            label = f"Z{r.train_idx}: NONE ov={r.overlap}"
        else:
            pos_a = int(round((r.s_norm or 0.0) * 100))          # Anfang->Ende
            pos_b = int(round((r.s_norm_rev or 0.0) * 100))      # Ende->Anfang
            lat = int(round(r.lateral_px or 0.0))
            label = f"Z{r.train_idx}: {r.track_id} A->E={pos_a}% E->A={pos_b}% lat={lat}px"

        cv2.putText(out, label, (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return out


# =========================
# MAIN

def main():
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise RuntimeError(f"Image not found: {IMAGE_PATH}")

    if FORCE_RESIZE_TO_CANVAS:
        canvas = cv2.resize(frame, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    else:
        canvas = frame

    shape_hw, tracks, track_masks = load_tracks_and_masks(TRACKMAP_JSON)

    # ====================================
    # HIER: YOLO OUTPUT EINFÜGEN (in Canvas-Koordinaten!)
    # Wenn kein Zug: obbs_canvas = []
    
    obbs_canvas = [
        [(1050, 60), (1220, 40), (1240, 90), (1070, 110)],
        [(980, 250), (1210, 230), (1230, 290), (1000, 310)],
    ]

    if not obbs_canvas:
        print("Kein Zug erkannt.")
        return

    results = process_trains(obbs_canvas, tracks, track_masks, shape_hw)

    for r in results:
        if r.track_id is None:
            print(f"Zug {r.train_idx}: kein Gleis (overlap={r.overlap})")
        else:
            print(
                f"Zug {r.train_idx}: Gleis={r.track_id} | "
                f"A->E={r.s_norm:.3f} ({r.s_norm*100:.1f}%) | "
                f"E->A={r.s_norm_rev:.3f} ({r.s_norm_rev*100:.1f}%) | "
                f"s_px={r.s_px:.1f} | lateral_px={r.lateral_px:.1f} | overlap={r.overlap}"
            )

    vis = draw_debug(canvas, tracks, results)
    cv2.imwrite("debug_track_position_var1.png", vis)
    print("Wrote debug_track_position_var1.png")

    cv2.imshow("Track Position (Var1)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()