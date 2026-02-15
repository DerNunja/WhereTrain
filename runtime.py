import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import cv2
import numpy as np

from Mapping.helper_section_tool import (
    autodetect_dictionary,
    make_detector,
    detect_markers,
    compute_section_src_pts_center,  
    warp,
)

Point = Tuple[int, int]


IMAGE_PATH = "Mapping/Pictures/different.jpg"
TRACKMAP_DIR = "Mapping/Sections"       # *__trackmap.json
SHOW_DEBUG = True

# if you want to ignore weak overlap
MIN_OVERLAP_PX = 50

WIN_W, WIN_H = 1280, 720
def setup_window(name: str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)     
    cv2.resizeWindow(name, WIN_W, WIN_H)   


def parse_section_from_trackmap_filename(path: str) -> Dict[str, Any]:
    """
    Expect something like:
      abschnitt_1__ids=TL1_TR7_BR5_BL2__1280x640__dict=DICT_ARUCO_ORIGINAL__trackmap.json
    """
    base = os.path.basename(path)

    m = re.match(
        r"(?P<section>.+?)__ids=(?P<ids>TL\d+_TR\d+_BR\d+_BL\d+)__(?P<w>\d+)x(?P<h>\d+).*__trackmap\.json$",
        base
    )
    if not m:
        raise ValueError(f"Trackmap filename does not match pattern: {base}")

    section_id = m.group("section")
    ids_str = m.group("ids")
    w, h = int(m.group("w")), int(m.group("h"))

    corner_ids: Dict[str, int] = {}
    for part in ids_str.split("_"):
        mm = re.match(r"(TL|TR|BR|BL)(\d+)", part)
        if mm:
            corner_ids[mm.group(1)] = int(mm.group(2))

    if any(k not in corner_ids for k in ["TL", "TR", "BR", "BL"]):
        raise ValueError(f"Missing corner IDs in filename: {base}")

    return {
        "section_id": section_id,
        "corner_ids": corner_ids,
        "canvas": (w, h),
        "raw": base,
    }



@dataclass
class Track:
    track_id: str
    polyline: List[Point]
    band: List[Point]


@dataclass
class Section:
    section_id: str
    canvas: Tuple[int, int]
    corner_ids: Dict[str, int]
    tracks: List[Track]
    json_path: str


def load_sections(trackmap_dir: str) -> List[Section]:
    sections: List[Section] = []

    for fn in sorted(os.listdir(trackmap_dir)):
        if not fn.endswith("__trackmap.json"):
            continue

        path = os.path.join(trackmap_dir, fn)
        info = parse_section_from_trackmap_filename(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracks: List[Track] = []
        for tr in data.get("tracks", []):
            polyline = [(int(p[0]), int(p[1])) for p in tr.get("polyline", [])]
            band = [(int(p[0]), int(p[1])) for p in tr.get("band", [])]
            tracks.append(Track(track_id=str(tr["track_id"]), polyline=polyline, band=band))

        sections.append(
            Section(
                section_id=info["section_id"],
                canvas=info["canvas"],
                corner_ids=info["corner_ids"],
                tracks=tracks,
                json_path=path,
            )
        )

    return sections



def _to_cv_poly(pts: List[Point]) -> Optional[np.ndarray]:
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr.reshape((-1, 1, 2))


def draw_tracks_overlay(warped: np.ndarray, tracks: List[Track]) -> np.ndarray:
    out = warped.copy()

    for tr in tracks:
        # band polygon (green filled)
        band = _to_cv_poly(tr.band)
        if band is not None and len(tr.band) >= 3:
            overlay = out.copy()
            cv2.fillPoly(overlay, [band], (0, 255, 0))
            out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
            cv2.polylines(out, [band], True, (0, 255, 0), 2)

        # center polyline (cyan)
        pl = _to_cv_poly(tr.polyline)
        if pl is not None and len(tr.polyline) >= 2:
            cv2.polylines(out, [pl], False, (255, 255, 0), 2)

        if tr.polyline:
            cv2.putText(out, tr.track_id, tr.polyline[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    return out


def draw_bboxes(img: np.ndarray, bboxes: List[Tuple[int, int, int, int]], label="train") -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def polygon_to_mask(poly: List[Point], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) < 3:
        return mask
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def bbox_to_mask(bbox: Tuple[int, int, int, int], shape_hw: Tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))

    mask = np.zeros((h, w), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def overlap_area(bbox: Tuple[int, int, int, int], band_poly: List[Point], shape_hw: Tuple[int, int]) -> int:
    bm = bbox_to_mask(bbox, shape_hw)
    tm = polygon_to_mask(band_poly, shape_hw)
    inter = cv2.bitwise_and(bm, tm)
    return int(cv2.countNonZero(inter))


def assign_bbox_to_track(bbox: Tuple[int, int, int, int], tracks: List[Track], shape_hw: Tuple[int, int]) -> Tuple[Optional[str], int]:
    best_tid = None
    best_area = -1
    for tr in tracks:
        if len(tr.band) < 3:
            continue
        a = overlap_area(bbox, tr.band, shape_hw)
        if a > best_area:
            best_area = a
            best_tid = tr.track_id
    if best_area < MIN_OVERLAP_PX:
        return None, best_area
    return best_tid, best_area


# detection modell
def detect_trains_stub(warped_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Replace with YOLO etc.
    Return [(x1,y1,x2,y2), ...] in normalized image coordinates.
    """
    return []


def main():
    sections = load_sections(TRACKMAP_DIR)
    if not sections:
        raise RuntimeError(f"No *__trackmap.json files found in {TRACKMAP_DIR}")

    print(f"Loaded {len(sections)} sections:")
    for s in sections:
        print(f" - {s.section_id} canvas={s.canvas} ids={s.corner_ids} tracks={len(s.tracks)}")

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) autodetect dictionary (from helpers)
    count, dict_name, dict_id = autodetect_dictionary(gray)
    print(f"Autodict: {dict_name} (found markers: {count})")
    detector = make_detector(dict_id)

    corners, ids = detect_markers(gray, detector)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No markers detected in frame.")

    # 2) build id -> corners(4,2)
    id_to_corners: Dict[int, np.ndarray] = {}
    for i, mid in enumerate(ids.flatten().tolist()):
        id_to_corners[int(mid)] = corners[i].reshape(4, 2)

    # debug input
    if SHOW_DEBUG:
        vis_in = frame.copy()
        cv2.aruco.drawDetectedMarkers(vis_in, corners, ids)

        setup_window("Input (detected markers)")
        cv2.imshow("Input (detected markers)", vis_in)

    # 3) process each section
    for s in sections:
        src_pts = compute_section_src_pts_center(id_to_corners, s.corner_ids)
        if src_pts is None:
            print(f"[{s.section_id}] missing marker(s) in current frame → skip")
            continue

        warped, H = warp(frame, src_pts, s.canvas)

        # 4) run train detection on normalized image
        bboxes = detect_trains_stub(warped)

        # 5) assign boxes to tracks
        assignments = []
        shape_hw = (s.canvas[1], s.canvas[0])

        for bb in bboxes:
            tid, area = assign_bbox_to_track(bb, s.tracks, shape_hw)
            assignments.append((bb, tid, area))

        # 6) visualize
        out = draw_tracks_overlay(warped, s.tracks)
        out = draw_bboxes(out, bboxes, "train")

        # print results
        if assignments:
            for (bb, tid, area) in assignments:
                print(f"[{s.section_id}] bbox={bb} -> track={tid} overlap_px={area}")
        else:
            print(f"[{s.section_id}] no trains detected (stub)")

        if SHOW_DEBUG:
            win_name = f"Warped+Tracks [{s.section_id}]"
            setup_window(win_name)
            cv2.imshow(win_name, out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
