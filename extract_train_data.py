import os
import glob
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np

from Mapping.helper_section_tool import (
    autodetect_dictionary,
    make_detector,
    detect_markers,
    compute_section_src_pts_center,
    warp,
)

VIDEO_GLOB = "data/TrainVid*.mp4"
OUT_DIR = "data/normalized_images"
EVERY_N_FRAMES = 10
MAX_FRAMES = None
USE_MARKER_CENTER = True

# DEBUG
DEBUG_MASK = False                  # Fenster/Debug aktivieren
DEBUG_SAVE_MASK_FILES = False       # Maske als Datei neben Bild speichern
DEBUG_SHOW_ONLY_WHEN_SAVED = False  # nur zeigen, wenn Bild gespeichert würde
DEBUG_WINDOW_W, DEBUG_WINDOW_H = 1400, 800

SECTIONS = [
    {
        "section_id": "abschnitt_1",
        "canvas": (1280, 640),
        "corner_ids": {"TL": 9, "TR": 3, "BR": 4, "BL": 21},
    },
    {
        "section_id": "abschnitt_2",
        "canvas": (1280, 640),
        "corner_ids": {"TL": 3, "TR": 5, "BR": 16, "BL": 4},
    },
]


Point = Tuple[int, int]


@dataclass
class Section:
    section_id: str
    canvas: Tuple[int, int]
    corner_ids: Dict[str, int]
    last_H: Optional[np.ndarray] = None


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def build_id_to_corners(corners, ids) -> Dict[int, np.ndarray]:
    id_to_corners: Dict[int, np.ndarray] = {}
    if ids is None:
        return id_to_corners
    for i, mid in enumerate(ids.flatten().tolist()):
        id_to_corners[int(mid)] = corners[i].reshape(4, 2)
    return id_to_corners


def setup_window(name: str, w: int, h: int):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)


def has_train_colors(
    bgr: np.ndarray,
    min_pixels: int = 300,
    min_ratio: float = 0.0003
):
    """
    Returns:
      ok(bool), mask(uint8 0/255), count(int), needed(int)
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Grün
    g_lo = np.array([35, 90, 60], dtype=np.uint8)
    g_hi = np.array([85, 255, 200], dtype=np.uint8)

    # Gelb
    y_lo = np.array([100, 100, 20], dtype=np.uint8)
    y_hi = np.array([255, 255, 40], dtype=np.uint8)

    mask_g = cv2.inRange(hsv, g_lo, g_hi)
    mask_y = cv2.inRange(hsv, y_lo, y_hi)
    mask = cv2.bitwise_or(mask_g, mask_y)

    count = int(cv2.countNonZero(mask))
    h, w = bgr.shape[:2]
    needed = max(min_pixels, int(h * w * min_ratio))
    ok = count >= needed
    return ok, mask, count, needed


def mask_overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Visual overlay: mask pixels are highlighted in red.
    """
    out = bgr.copy()
    red = np.zeros_like(out)
    red[:, :, 2] = 255  # red channel
    alpha = 0.35
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return out


def process_video(video_path: str) -> int:
    video_stem = os.path.splitext(os.path.basename(video_path))[0]

    sections: List[Section] = [
        Section(section_id=s["section_id"], canvas=s["canvas"], corner_ids=s["corner_ids"])
        for s in SECTIONS
    ]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, first = cap.read()
    if not ok:
        cap.release()
        return 0

    gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    count, dict_name, dict_id = autodetect_dictionary(gray0)
    print(f"[{video_stem}] Dict: {dict_name} ({count} markers)")
    detector = make_detector(dict_id)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if DEBUG_MASK:
        setup_window("DEBUG | warped", DEBUG_WINDOW_W, DEBUG_WINDOW_H)
        setup_window("DEBUG | mask", DEBUG_WINDOW_W, DEBUG_WINDOW_H)
        setup_window("DEBUG | overlay", DEBUG_WINDOW_W, DEBUG_WINDOW_H)

    frame_idx = 0
    saved_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
            break

        if frame_idx % EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(gray, detector)
        id_to_corners = build_id_to_corners(corners, ids)

        for s in sections:
            src_pts = compute_section_src_pts_center(id_to_corners, s.corner_ids)

            if src_pts is not None:
                warped, H = warp(frame, src_pts, s.canvas)
                s.last_H = H
            else:
                if s.last_H is None:
                    continue
                w, h = s.canvas
                warped = cv2.warpPerspective(frame, s.last_H, (w, h))

            ok_color, mask, count_px, needed_px = has_train_colors(warped)

            # Optional debug display
            if DEBUG_MASK and (ok_color if DEBUG_SHOW_ONLY_WHEN_SAVED else True):
                overlay = mask_overlay(warped, mask)
                hud = f"{video_stem} | {s.section_id} | f={frame_idx} | mask_px={count_px} needed={needed_px} | SAVE={ok_color}"
                cv2.putText(overlay, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("DEBUG | warped", warped)
                cv2.imshow("DEBUG | mask", mask)
                cv2.imshow("DEBUG | overlay", overlay)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    return saved_total

            if not ok_color:
                continue

            out_name = f"{video_stem}__{s.section_id}__f={frame_idx:06d}.png"
            out_path = os.path.join(OUT_DIR, out_name)
            cv2.imwrite(out_path, warped)
            saved_total += 1

            if DEBUG_SAVE_MASK_FILES:
                mask_name = f"{video_stem}__{s.section_id}__f={frame_idx:06d}__mask.png"
                mask_path = os.path.join(OUT_DIR, mask_name)
                cv2.imwrite(mask_path, mask)

                overlay_name = f"{video_stem}__{s.section_id}__f={frame_idx:06d}__overlay.png"
                overlay_path = os.path.join(OUT_DIR, overlay_name)
                cv2.imwrite(overlay_path, mask_overlay(warped, mask))

        frame_idx += 1

    cap.release()
    if DEBUG_MASK:
        cv2.destroyAllWindows()

    print(f"[{video_stem}] saved {saved_total} images")
    return saved_total


def main():
    ensure_dir(OUT_DIR)

    videos = sorted(glob.glob(VIDEO_GLOB))
    if not videos:
        raise RuntimeError("No TrainVid*.mp4 found")

    grand_total = 0
    for v in videos:
        grand_total += process_video(v)

    print(f"\nDONE. Total saved images: {grand_total}")
    print(f"All images are in: {OUT_DIR}")


if __name__ == "__main__":
    main()

