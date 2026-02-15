import cv2
import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
from helper_section_tool import (
    make_detector, detect_markers, autodetect_dictionary,
    compute_section_src_pts, compute_section_src_pts_center,
    draw_hud, warp, nearest_marker_id, build_output_name
)

# CONFIG
USE_WEBCAM = False
WEBCAM_INDEX = 0
IMAGE_PATH = "Pictures/Base.jpg"
scale = 0.5
use_marker_center = True

SECTIONS_DIR = "Sections"
OUTPUT_PREFIX = "out_sections"
OUTPUT_JSON = "sections.json"

DICT_CANDIDATES = [
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_APRILTAG_36h11", cv2.aruco.DICT_APRILTAG_36h11),
]

CANVAS_PRESETS: Dict[str, Tuple[Tuple[int, int], str]] = {
    "1": ((1280, 640),  "2:1 (1280x640) default"),
    "2": ((1024, 1024), "1:1 (1024x1024)"),
    "3": ((1280, 720),  "16:9 (1280x720)"),
    "4": ((1920, 1080), "16:9 (1920x1080)"),
    "5": ((640, 640),   "1:1 small (640x640)"),
    "6": ((1600, 800),  "2:1 (1600x800)"),
}
DEFAULT_PRESET_KEY = "1"

Point = Tuple[int, int]


@dataclass
class SectionDef:
    section_id: str
    canvas_w: int
    canvas_h: int
    corner_ids: Dict[str, Optional[int]]  # TL/TR/BR/BL


class MultiNormalizeTool:
    def __init__(self):
        self.output_prefix = OUTPUT_PREFIX

        self.sections: List[SectionDef] = []
        self.current_idx: int = -1

        self.detector: Optional[cv2.aruco.ArucoDetector] = None
        self.autodict_name: Optional[str] = None
        self.current_dict_id: Optional[int] = None

        self.last_corners = None
        self.last_ids = None
        self.last_frame = None

        self.await_corner: Optional[str] = None

        self.await_preset_for_new_section: bool = False
        self._pending_new_section_name: Optional[str] = None

    def _ensure_default_section(self):
        if not self.sections:
            (w, h), _label = CANVAS_PRESETS[DEFAULT_PRESET_KEY]
            self.sections.append(
                SectionDef(
                    section_id="abschnitt_1",
                    canvas_w=w,
                    canvas_h=h,
                    corner_ids={"TL": None, "TR": None, "BR": None, "BL": None},
                )
            )
            self.current_idx = 0

    def _cur(self) -> SectionDef:
        self._ensure_default_section()
        return self.sections[self.current_idx]

    def _print_presets(self):
        print("Canvas presets:")
        for k, (wh, label) in CANVAS_PRESETS.items():
            print(f"  {k}: {label}")
        print(f"Default preset: {DEFAULT_PRESET_KEY}")

    def add_section(self):
        name = input("New section_id (e.g. abschnitt_2): ").strip()
        if not name:
            print("Cancelled.")
            return

        self.await_preset_for_new_section = True
        self._pending_new_section_name = name
        self._print_presets()
        print("Choose preset by pressing 1..6 in the OpenCV window...")

    def _finalize_add_section_with_preset(self, preset_key: str):
        if not self.await_preset_for_new_section or not self._pending_new_section_name:
            return
        if preset_key not in CANVAS_PRESETS:
            return

        (w, h), label = CANVAS_PRESETS[preset_key]
        name = self._pending_new_section_name

        self.sections.append(
            SectionDef(
                section_id=name,
                canvas_w=w,
                canvas_h=h,
                corner_ids={"TL": None, "TR": None, "BR": None, "BL": None},
            )
        )
        self.current_idx = len(self.sections) - 1
        self.await_corner = None

        print(f"Added section '{name}' with canvas {w}x{h} ({label})")

        self.await_preset_for_new_section = False
        self._pending_new_section_name = None

    def next_section(self, direction: int):
        self._ensure_default_section()
        self.current_idx = (self.current_idx + direction) % len(self.sections)
        self.await_corner = None
        print(f"Active section: {self._cur().section_id}")

    def reset_current_corners(self):
        s = self._cur()
        s.corner_ids = {"TL": None, "TR": None, "BR": None, "BL": None}
        self.await_corner = None
        print(f"Reset corners for {s.section_id}")

    def save_json(self, path=OUTPUT_JSON):
        payload = {
            "image_path": IMAGE_PATH,
            "scale": scale,
            "aruco_dict": self.autodict_name,
            "canvas_presets": {k: {"w": wh[0][0], "h": wh[0][1], "label": wh[1]} for k, wh in CANVAS_PRESETS.items()},
            "sections": [asdict(s) for s in self.sections],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved sections JSON: {path}")

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.last_ids is None or self.last_corners is None:
            return
        if self.await_corner is None:
            return

        mid = nearest_marker_id((x, y), self.last_corners, self.last_ids)
        if mid is None:
            return

        s = self._cur()
        s.corner_ids[self.await_corner] = mid
        print(f"[{s.section_id}] Set {self.await_corner} marker ID = {mid}")
        self.await_corner = None

    def _compute_src_pts_for_section(self, s: SectionDef, id_to_corners: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        if not all(s.corner_ids[k] is not None for k in ["TL", "TR", "BR", "BL"]):
            return None
        corner_ids_int = {k: int(v) for k, v in s.corner_ids.items()}

        if use_marker_center == True:
            return compute_section_src_pts_center(id_to_corners, corner_ids_int)
        else:
            return compute_section_src_pts(id_to_corners, corner_ids_int)

    def export_section(self, frame: np.ndarray, id_to_corners: Dict[int, np.ndarray], s: SectionDef):
        src_pts = self._compute_src_pts_for_section(s, id_to_corners)
        if src_pts is None:
            print(f"[{s.section_id}] Cannot export (need 4 IDs + visible markers).")
            return
        canvas = (s.canvas_w, s.canvas_h)

        os.makedirs(SECTIONS_DIR, exist_ok=True)

        warped_img, _H = warp(frame, src_pts, canvas)

        name_png = build_output_name(
            s.section_id,
            s.corner_ids,
            (s.canvas_w, s.canvas_h),
            self.autodict_name,
            ".png"
        )

        cv2.imwrite(os.path.join(SECTIONS_DIR, name_png), warped_img)

        print(f"[{s.section_id}] Saved to '{SECTIONS_DIR}/'")

    def export_all(self, frame: np.ndarray, id_to_corners: Dict[int, np.ndarray]):
        for s in self.sections:
            self.export_section(frame, id_to_corners, s)

    # main loop
    def run(self):
        # load frame
        if USE_WEBCAM:
            cap = cv2.VideoCapture(WEBCAM_INDEX)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open webcam index {WEBCAM_INDEX}")
        else:
            cap = None
            frame = cv2.imread(IMAGE_PATH)
            if frame is None:
                raise RuntimeError(f"Cannot read image: {IMAGE_PATH}")
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self.last_frame = frame

        self._ensure_default_section()

        cv2.namedWindow("Input", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Input", self.on_mouse)

        print("Controls:")
        print("  D: autodetect dict | A: add section (then press 1..6 preset key in window)")
        print("  [ / ]: prev/next section")
        print("  1/2/3/4 then click: set TL/TR/BR/BL marker ID (unless preset selection is active)")
        print("  R: reset current section corners")
        print("  X: export current section | E: export all")
        print("  J: save sections.json | Q/ESC: quit")
        print("  P: print presets to terminal")

        while True:
            if USE_WEBCAM:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read webcam.")
                    break
                self.last_frame = frame
            else:
                frame = self.last_frame

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # pick dict if missing
            if self.current_dict_id is None:
                count, name, dict_id = autodetect_dictionary(gray)
                self.autodict_name = name
                self.current_dict_id = dict_id
                self.detector = make_detector(dict_id)
                print(f"Autodetected dictionary: {name} (markers: {count})")

            corners, ids = detect_markers(gray, self.detector)
            self.last_corners, self.last_ids = corners, ids

            vis = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            # id->corners
            id_to_corners: Dict[int, np.ndarray] = {}
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    id_to_corners[int(mid)] = corners[i].reshape(4, 2)

            # draw all saved section quads (green), active in red
            for idx, s in enumerate(self.sections):
                src_pts = self._compute_src_pts_for_section(s, id_to_corners)
                if src_pts is None:
                    continue
                color = (0, 0, 255) if idx == self.current_idx else (0, 255, 0)
                cv2.polylines(vis, [src_pts.astype(int)], True, color, 3)
                cv2.putText(vis, s.section_id, tuple(src_pts.astype(int)[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cur = self._cur()
            found_ids = [] if ids is None else ids.flatten().tolist()
            status = "PRESET SELECT: press 1..6" if self.await_preset_for_new_section else "normal"
            hud = [
                f"Dict: {self.autodict_name} | Found: {len(found_ids)} | IDs: {found_ids}",
                f"Mode: {status}",
                f"Active: {cur.section_id} [{self.current_idx+1}/{len(self.sections)}] | Canvas: {cur.canvas_w}x{cur.canvas_h}",
                f"Corners: TL={cur.corner_ids['TL']} TR={cur.corner_ids['TR']} BR={cur.corner_ids['BR']} BL={cur.corner_ids['BL']}",
                f"Click-Assign: {self.await_corner if self.await_corner else '-'} (1/2/3/4 then click)",
                "Keys: D autodict | A add | [ ] switch | R reset | X export | E exportAll | J saveJSON | Q quit | P presets",
            ]
            draw_hud(vis, hud)

            cv2.imshow("Input", vis)

            # preview warped for active section only
            active_src = self._compute_src_pts_for_section(cur, id_to_corners)
            if active_src is not None:
                warped_img, _ = warp(frame, active_src, (cur.canvas_w, cur.canvas_h))
                cv2.imshow("Warped", warped_img)
            else:
                blank = np.zeros((cur.canvas_h, cur.canvas_w, 3), dtype=np.uint8)
                cv2.putText(blank, "Active section: need 4 corner IDs + all visible", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Warped", blank)

            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), 27):
                break

            if key in (ord('p'), ord('P')):
                self._print_presets()

            if key in (ord('d'), ord('D')):
                count, name, dict_id = autodetect_dictionary(gray)
                self.autodict_name = name
                self.current_dict_id = dict_id
                self.detector = make_detector(dict_id)
                print(f"Autodetected dictionary: {name} (markers: {count})")

            if key in (ord('a'), ord('A')):
                self.add_section()

            if key == ord('['):
                self.next_section(-1)
            if key == ord(']'):
                self.next_section(+1)

            ch = chr(key) if key != 255 else ""
            if self.await_preset_for_new_section and ch in CANVAS_PRESETS:
                self._finalize_add_section_with_preset(ch)
                continue

            if key == ord('1'):
                self.await_corner = "TL"
            if key == ord('2'):
                self.await_corner = "TR"
            if key == ord('3'):
                self.await_corner = "BR"
            if key == ord('4'):
                self.await_corner = "BL"

            if key in (ord('r'), ord('R')):
                self.reset_current_corners()

            if key in (ord('j'), ord('J')):
                self.save_json()

            if key in (ord('x'), ord('X')):
                self.export_section(frame, id_to_corners, self._cur())

            if key in (ord('e'), ord('E')):
                self.export_all(frame, id_to_corners)

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MultiNormalizeTool().run()
