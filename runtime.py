import os
import json
import time
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

from Mapping.helper_map_tool import parse_section_from_filename

from Detection.Color_detcion.detection_with_color import detect_by_color

Point = Tuple[int, int]

USE_IMAGE = False
USE_VIDEO = True
USE_WEBCAM = False
SHOW_DEBUG = True

WEBCAM_INDEX = 0    #live
IMAGE_PATH = "Mapping/Pictures/different.jpg"
VIDEO_PATH = "data/TestVid3.mp4"
TRACKMAP_DIR = "Mapping/Sections"       # *__trackmap.json

# optional: Video-Performance
PROCESS_EVERY_NTH_FRAME = 1   # 1 = jeden Frame, 2 = jeden 2ten, ...
H_TIMEOUT_SEC = 8.0   # solange (in Sekunden) darf alte H genutzt werden, wenn Marker fehlen

MIN_OVERLAP_PX = 50
WIN_W, WIN_H = 1280, 720


def setup_window(name: str):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)     
    cv2.resizeWindow(name, WIN_W, WIN_H)   


def parse_section_from_trackmap_filename(path: str) -> Dict[str, Any]:
    info = parse_section_from_filename(path)
    if not info.get("corner_ids") or not info.get("canvas"):
        raise ValueError(f"Missing ids/canvas in filename: {os.path.basename(path)}")
    return info


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

#------------------------------------------------Code Firas ------------------------------------------------
PtF = Tuple[float, float]
Point = Tuple[int, int]

def polygon_center(poly: List[Point]) -> PtF:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (float(sum(xs)) / len(xs), float(sum(ys)) / len(ys))

def bbox_center(bbox: Tuple[int, int, int, int]) -> PtF:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def project_point_to_segment(p: PtF, a: PtF, b: PtF):
    """
    Projektion eines Punktes p auf Segment a-b.
    Returns: t in [0,1], proj point q, squared distance
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
    Returns (s_px, s_norm, lateral_px)
    s_px: Distanz entlang Polyline bis zur Projektion
    s_norm: s_px / total_len (0..1)
    lateral_px: seitlicher Abstand zur Mittellinie
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
#------------------------------------------------Code Firas ------------------------------------------------

# detection modell
def detect_trains_stub(warped_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    return detect_by_color(warped_bgr, min_area=400, morph_kernel=5, morph_iters=2)


def warp_with_H(frame_bgr: np.ndarray, H: np.ndarray, canvas: Tuple[int, int]) -> np.ndarray:
    """Warp via cv2.warpPerspective direkt mit gegebener Homography."""
    w, h = canvas 
    return cv2.warpPerspective(frame_bgr, H, (w, h))


def main():
    sections = load_sections(TRACKMAP_DIR)
    if not sections:
        raise RuntimeError(f"No *__trackmap.json files found in {TRACKMAP_DIR}")

    print(f"Loaded {len(sections)} sections:")
    for s in sections:
        print(f" - {s.section_id} canvas={s.canvas} ids={s.corner_ids} tracks={len(s.tracks)}")

    cap = None
    single_image = None

    if USE_IMAGE:
        single_image = cv2.imread(IMAGE_PATH)
        if single_image is None:
            raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")
    else:
        if USE_WEBCAM:
            cap = cv2.VideoCapture(WEBCAM_INDEX)
        else:
            cap = cv2.VideoCapture(VIDEO_PATH)

        if not cap.isOpened():
            src = f"webcam index {WEBCAM_INDEX}" if USE_WEBCAM else VIDEO_PATH
            raise RuntimeError(f"Cannot open video source: {src}")

    if SHOW_DEBUG:
        setup_window("Input (detected markers)")

    frame_idx = 0
    prev_frame_ts = time.perf_counter()
    fps_smoothed = 0.0
    last_H: Dict[str, np.ndarray] = {}
    last_H_time: Dict[str, float] = {}
    paused = False

    while True:
        if USE_IMAGE:
            frame = single_image.copy()
            ok = True
        else:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):  # unpause
                    paused = False
                elif key == 27:      # ESC
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"frame_paused_{frame_idx:06d}.png", frame)
                    print(f"Saved frame_paused_{frame_idx:06d}.png")
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                break  # end of video

        frame_idx += 1
        now_perf = time.perf_counter()
        dt = now_perf - prev_frame_ts
        prev_frame_ts = now_perf
        if dt > 0:
            fps_current = 1.0 / dt
            fps_smoothed = fps_current if fps_smoothed <= 0 else (0.9 * fps_smoothed + 0.1 * fps_current)

        if PROCESS_EVERY_NTH_FRAME > 1 and (frame_idx % PROCESS_EVERY_NTH_FRAME) != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1) autodetect dictionary (from helpers)
        count, dict_name, dict_id = autodetect_dictionary(gray)
        detector = make_detector(dict_id)

        corners, ids = detect_markers(gray, detector)

        # Debug input
        if SHOW_DEBUG:
            vis_in = frame.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis_in, corners, ids)
            cv2.putText(vis_in, f"FPS: {fps_smoothed:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Input (detected markers)", vis_in)

        have_markers = (ids is not None and len(ids) > 0)

        # 2) build id -> corners(4,2)
        id_to_corners: Dict[int, np.ndarray] = {}
        if have_markers:
            for i, mid in enumerate(ids.flatten().tolist()):
                id_to_corners[int(mid)] = corners[i].reshape(4, 2)

        # 3) process each section
        now = time.time()

        for s in sections:
            H_use = None
            warped = None

            # Versuch: neue Homographie aus aktuellen Markern
            if have_markers:
                src_pts = compute_section_src_pts_center(id_to_corners, s.corner_ids)
                if src_pts is not None:
                    warped, H_new = warp(frame, src_pts, s.canvas)

                    # Cache aktualisieren
                    last_H[s.section_id] = H_new
                    last_H_time[s.section_id] = now
                    H_use = H_new

            # Fallback: letzte Homographie nutzen, wenn frisch genug
            if warped is None:
                H_cached = last_H.get(s.section_id)
                t_cached = last_H_time.get(s.section_id, -1.0)

                if H_cached is not None and (now - t_cached) <= H_TIMEOUT_SEC:
                    warped = warp_with_H(frame, H_cached, s.canvas)
                    H_use = H_cached
                else:
                    # kein gültiger Warp möglich
                    continue

            # 4) run train detection on normalized image
            bboxes = detect_trains_stub(warped)

            # 5) assign boxes to tracks
            shape_hw = (s.canvas[1], s.canvas[0])
            assignments = []

            track_by_id = {tr.track_id: tr for tr in s.tracks}

            for bb in bboxes:
                tid, area = assign_bbox_to_track(bb, s.tracks, shape_hw)

                cx, cy = bbox_center(bb)

                if tid is not None and tid in track_by_id:
                    s_px, s_norm, lateral = position_on_track((cx, cy), track_by_id[tid].polyline)
                else:
                    s_px, s_norm, lateral = None, None, None

                print(f"[{s.section_id}] bb={bb} -> track={tid} ov={area} s={s_norm} lat={lateral}")


            # 6) visualize
            out = draw_tracks_overlay(warped, s.tracks)
            out = draw_bboxes(out, bboxes, "train")

            if SHOW_DEBUG:
                win_name = f"Warped+Tracks [{s.section_id}]"
                setup_window(win_name)

                # kleine Statusanzeige: ob H neu oder cached
                status = "H:NEW" if (have_markers and s.section_id in last_H_time and abs(last_H_time[s.section_id] - now) < 1e-3) else "H:CACHED"
                cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(out, f"FPS: {fps_smoothed:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(win_name, out)

            for (bb, tid, area) in assignments:
                if tid is not None:
                    print(f"[frame {frame_idx:06d}][{s.section_id}] bbox={bb} -> track={tid} overlap_px={area}")


        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC
            break
        elif key == ord(' '):  # Pause
            paused = True
        elif key == ord('s'):  # Save current input frame
            cv2.imwrite(f"frame_{frame_idx:06d}.png", frame)
            print(f"Saved frame_{frame_idx:06d}.png")

        if USE_IMAGE:
            # For images we process exactly once
            cv2.waitKey(0)
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
