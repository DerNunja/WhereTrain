import cv2
import json
import os
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from helper_map_tool import parse_section_from_filename, default_output_json_path, to_py_points, track_sort_key, polyline_to_band, parse_track_number, next_free_track_id


IMAGE_PATH = "Sections/ab2__ids=TL9_TR10_BR8_BL5__1280x640__dict=DICT_ARUCO_ORIGINAL.png"
WINDOW_NAME = "Track Mapping Tool"

DRAW_RADIUS = 4
LINE_THICKNESS = 2

TRACK_WIDTH_PX = 60  

Point = Tuple[int, int]


@dataclass
class Track:
    track_id: str
    polyline: List[Point]
    band: List[Point]


@dataclass
class Zone:
    zone_id: str
    polygon: List[Point]



class MapperTool:
    def __init__(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.image_path = image_path
        self.info = parse_section_from_filename(image_path)
        self.section_id = self.info["section_id"]
        self.output_json = default_output_json_path(image_path)

        self.img_orig = cv2.imread(image_path)
        if self.img_orig is None:
            raise RuntimeError("Failed to load image (cv2.imread returned None).")

        self.h, self.w = self.img_orig.shape[:2]
        self.img = self.img_orig.copy()

        self.mode = "track"  
        self.current_points: List[Point] = []

        self.tracks: Dict[str, Track] = {}
        self.zones: Dict[str, Zone] = {}

        self.track_width_px = TRACK_WIDTH_PX
        self.current_track_id = "G1"
        self.current_zone_id = "EXIT_A"

        self._dirty = True

    def _to_cv_poly(self, pts: List[Point]) -> Optional[np.ndarray]:
        if not pts:
            return None
        arr = np.asarray(pts, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        return arr.reshape((-1, 1, 2))

    def _redraw(self):
        self.img = self.img_orig.copy()

        # draw saved tracks
        for tid, tr in sorted(self.tracks.items(), key=lambda kv: track_sort_key(kv[0])):
            band = self._to_cv_poly(tr.band)
            if band is not None and len(tr.band) >= 3:
                overlay = self.img.copy()
                cv2.fillPoly(overlay, [band], (0, 255, 0))
                self.img = cv2.addWeighted(overlay, 0.20, self.img, 0.80, 0)
                cv2.polylines(self.img, [band], True, (0, 255, 0), LINE_THICKNESS)

            pl = self._to_cv_poly(tr.polyline)
            if pl is not None and len(tr.polyline) >= 2:
                cv2.polylines(self.img, [pl], False, (255, 255, 0), LINE_THICKNESS)

            if tr.polyline:
                cv2.putText(self.img, tid, tr.polyline[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # draw saved zones
        for zid, zn in self.zones.items():
            zpoly = self._to_cv_poly(zn.polygon)
            if zpoly is not None and len(zn.polygon) >= 3:
                cv2.polylines(self.img, [zpoly], True, (0, 165, 255), LINE_THICKNESS)
                cv2.putText(self.img, zid, zn.polygon[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # draw current object being drawn
        if self.current_points:
            if self.mode == "track":
                color = (255, 255, 0)
                closed = False
            else:
                color = (0, 165, 255)
                closed = len(self.current_points) >= 3

            for p in self.current_points:
                cv2.circle(self.img, p, DRAW_RADIUS, color, -1)

            cur = self._to_cv_poly(self.current_points)
            if cur is not None and len(self.current_points) >= 2:
                cv2.polylines(self.img, [cur], closed, color, LINE_THICKNESS)

            if self.mode == "track" and len(self.current_points) >= 2:
                band_pts = polyline_to_band(self.current_points, self.track_width_px)
                band = self._to_cv_poly(band_pts)
                if band is not None and len(band_pts) >= 3:
                    cv2.polylines(self.img, [band], True, (0, 255, 0), 2)

        # HUD
        hud_lines = [
            f"Section: {self.section_id} | Mode: {self.mode} | TrackID: {self.current_track_id} | Tracks: {len(self.tracks)}",
            f"Track width: {self.track_width_px}px | Image: {os.path.basename(self.image_path)} ({self.w}x{self.h})",
            f"Save: {os.path.basename(self.output_json)}",
            "Keys: [T]=track  [Z]=zone  [ENTER]=finish / set ID  [N]=clear current  [BACKSPACE]=undo",
            "      [.]=next track  [,]=prev track  [+/-]=width  [DEL]=delete current track  [S]=save  [Q]=quit",
            "ID input: type digits + ENTER (e.g. 3 -> G3). (If no typing: ENTER commits points)",
        ]
        y = 25
        for line in hud_lines:
            cv2.putText(self.img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            y += 22

        self._dirty = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((int(x), int(y)))
            self._dirty = True

    def commit_current(self):
        pts = self.current_points

        if self.mode == "track":
            if len(pts) < 2:
                print("Track polyline needs at least 2 points.")
                return

            band = polyline_to_band(pts, self.track_width_px)
            self.tracks[self.current_track_id] = Track(
                track_id=self.current_track_id,
                polyline=pts.copy(),
                band=band,
            )
            print(f"Saved track: {self.current_track_id} (polyline {len(pts)} pts, band {len(band)} pts)")

            cur_num = parse_track_number(self.current_track_id) or 1
            self.current_track_id = next_free_track_id(set(self.tracks.keys()), start_from=cur_num + 1)
            print(f"Next track: {self.current_track_id}")

        else:  # zone
            if len(pts) < 3:
                print("Zone polygon needs at least 3 points.")
                return
            self.zones[self.current_zone_id] = Zone(zone_id=self.current_zone_id, polygon=pts.copy())
            print(f"Saved zone: {self.current_zone_id} ({len(pts)} pts)")

        self.current_points = []
        self._dirty = True

    def undo_point(self):
        if self.current_points:
            self.current_points.pop()
            self._dirty = True

    def clear_current(self):
        self.current_points = []
        self._dirty = True

    def delete_current_track(self):
        if self.mode != "track":
            return
        if self.current_track_id in self.tracks:
            del self.tracks[self.current_track_id]
            print(f"Deleted track: {self.current_track_id}")
            # choose next sensible id
            self.current_track_id = next_free_track_id(set(self.tracks.keys()), start_from=1)
            print(f"Now active: {self.current_track_id}")
            self._dirty = True

    def next_prev_track(self, direction: int):
        # direction: +1 next, -1 prev
        # collect numeric ids
        nums = sorted(parse_track_number(tid) for tid in self.tracks.keys() if parse_track_number(tid) is not None)
        cur = parse_track_number(self.current_track_id) or 1

        if not nums:
            # nothing yet -> just keep current
            self._dirty = True
            return

        if direction > 0:
            # next existing, otherwise next free after max
            higher = [n for n in nums if n > cur]
            if higher:
                self.current_track_id = f"G{higher[0]}"
            else:
                self.current_track_id = next_free_track_id(set(self.tracks.keys()), start_from=max(nums) + 1)
        else:
            lower = [n for n in nums if n < cur]
            if lower:
                self.current_track_id = f"G{lower[-1]}"
            else:
                self.current_track_id = f"G{nums[0]}"
        self.clear_current()
        print(f"Active track: {self.current_track_id}")

    def change_width(self, delta: int):
        self.track_width_px = int(np.clip(self.track_width_px + delta, 4, 500))
        print(f"Track width: {self.track_width_px}px")
        self._dirty = True

    def save_json(self, out_path: str):
        payload: Dict[str, Any] = {
            "section_id": self.section_id,
            "image_path": self.image_path,
            "canvas_size": {"w": self.w, "h": self.h},
            "track_width_px": self.track_width_px,
            "tracks": [],
            "zones": [],
            "meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "tool": "map_tool.py",
                "parsed_from_filename": self.info,
            }
        }

        for tid in sorted(self.tracks.keys(), key=track_sort_key):
            tr = self.tracks[tid]
            payload["tracks"].append({
                "track_id": tr.track_id,
                "polyline": to_py_points(tr.polyline),
                "band": to_py_points(tr.band),
            })

        for zid in sorted(self.zones.keys()):
            zn = self.zones[zid]
            payload["zones"].append({
                "zone_id": zn.zone_id,
                "polygon": zn.polygon,
            })

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Saved mapping to: {out_path}")

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        typed = ""

        while True:
            if self._dirty:
                self._redraw()

            cv2.imshow(WINDOW_NAME, self.img)
            key = cv2.waitKey(10) & 0xFF

            if key == 255:
                continue

            # quit
            if key in (ord('q'), ord('Q'), 27):
                break

            # modes
            if key in (ord('t'), ord('T')):
                self.mode = "track"
                self.clear_current()
                typed = ""
                continue

            if key in (ord('z'), ord('Z')):
                self.mode = "zone"
                self.clear_current()
                typed = ""
                continue

            # clear / undo
            if key in (ord('n'), ord('N')):
                self.clear_current()
                typed = ""
                continue

            if key in (8, 127):
                self.undo_point()
                typed = ""
                continue

            # next/prev track
            if key == ord('.'):
                self.next_prev_track(+1)
                typed = ""
                continue
            if key == ord(','):
                self.next_prev_track(-1)
                typed = ""
                continue

            # width
            if key in (ord('+'), ord('=')):  # '=' is + without shift on many keyboards
                self.change_width(+4)
                continue
            if key in (ord('-'), ord('_')):
                self.change_width(-4)
                continue

            if key in (83,):  # may not work everywhere; keep method for convenience
                self.delete_current_track()
                continue

            # commit or set ID
            if key in (13, 10):  # ENTER
                if typed:
                    if self.mode == "zone":
                        self.current_zone_id = typed.strip()
                        print(f"ZoneID set to: {self.current_zone_id}")
                    else:
                        t = typed.strip()
                        self.current_track_id = f"G{t}" if t.isdigit() else t
                        print(f"TrackID set to: {self.current_track_id}")
                    typed = ""
                    self._dirty = True
                else:
                    self.commit_current()
                continue

            # save
            if key in (ord('s'), ord('S')):
                self.save_json(self.output_json)
                continue

            # typing buffer
            ch = chr(key)
            if ch.isprintable():
                typed += ch

        cv2.destroyAllWindows()


if __name__ == "__main__":
    tool = MapperTool(IMAGE_PATH)
    tool.run()
