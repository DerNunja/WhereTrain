
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List

# CONFIG
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

# Canvas presets (key -> (w,h), label)
CANVAS_PRESETS: Dict[str, Tuple[Tuple[int, int], str]] = {
    "1": ((1280, 640),  "2:1 (1280x640) default"),
    "2": ((1024, 1024), "1:1 (1024x1024)"),
    "3": ((1280, 720),  "16:9 (1280x720)"),
    "4": ((1920, 1080), "16:9 (1920x1080)"),
    "5": ((640, 640),   "1:1 small (640x640)"),
    "6": ((640, 320),  "2:1 small(1600x800)"),
}
DEFAULT_PRESET_KEY = "1"

Point = Tuple[int, int]

def make_detector(dict_id: int) -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_markers(gray: np.ndarray, detector: cv2.aruco.ArucoDetector):
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids

def marker_center(marker_corners: np.ndarray) -> Point:
    pts = marker_corners.reshape(4, 2).astype(np.float32)
    c = pts.mean(axis=0)
    return (int(c[0]), int(c[1]))

def autodetect_dictionary(gray: np.ndarray):
    best = None  # (count, name, dict_id)
    for name, dict_id in DICT_CANDIDATES:
        detector = make_detector(dict_id)
        corners, ids = detect_markers(gray, detector)
        count = 0 if ids is None else len(ids)
        if best is None or count > best[0]:
            best = (count, name, dict_id)
    return best


def order_marker_corners(marker_corners: np.ndarray) -> Dict[str, Point]:
    pts = marker_corners.reshape(4, 2).astype(np.float32)  # OpenCV: TL, TR, BR, BL
    return {
        "TL": (int(pts[0][0]), int(pts[0][1])),
        "TR": (int(pts[1][0]), int(pts[1][1])),
        "BR": (int(pts[2][0]), int(pts[2][1])),
        "BL": (int(pts[3][0]), int(pts[3][1])),
    }


def compute_section_src_pts(id_to_corners: Dict[int, np.ndarray], corner_ids: Dict[str, int]) -> Optional[np.ndarray]:
    needed = [corner_ids["TL"], corner_ids["TR"], corner_ids["BR"], corner_ids["BL"]]
    if any(mid not in id_to_corners for mid in needed):
        return None

    mTL = order_marker_corners(id_to_corners[corner_ids["TL"]])["TL"]
    mTR = order_marker_corners(id_to_corners[corner_ids["TR"]])["TR"]
    mBR = order_marker_corners(id_to_corners[corner_ids["BR"]])["BR"]
    mBL = order_marker_corners(id_to_corners[corner_ids["BL"]])["BL"]

    return np.array([mTL, mTR, mBR, mBL], dtype=np.float32)

# besser um overlapping zu vermeiden
def compute_section_src_pts_center(
    id_to_corners: Dict[int, np.ndarray],
    corner_ids: Dict[str, int]
) -> Optional[np.ndarray]:
    """
    Compute the source points for a section of track given by its corners.
    """
    needed = [corner_ids["TL"], corner_ids["TR"], corner_ids["BR"], corner_ids["BL"]]
    if any(mid not in id_to_corners for mid in needed):
        return None

    cTL = marker_center(id_to_corners[corner_ids["TL"]])
    cTR = marker_center(id_to_corners[corner_ids["TR"]])
    cBR = marker_center(id_to_corners[corner_ids["BR"]])
    cBL = marker_center(id_to_corners[corner_ids["BL"]])

    return np.array([cTL, cTR, cBR, cBL], dtype=np.float32)


def warp(frame: np.ndarray, src_pts: np.ndarray, canvas: Tuple[int, int]):
    w, h = canvas
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, H, (w, h), flags=cv2.INTER_LINEAR)
    return warped, H


def draw_hud(img, lines: List[str], origin=(10, 25)):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 22


def nearest_marker_id(click: Point, corners: list, ids: np.ndarray) -> Optional[int]:
    if ids is None:
        return None
    best_id = None
    best_dist = 1e18
    cx, cy = click
    for i, mid in enumerate(ids.flatten().tolist()):
        pts = corners[i].reshape(4, 2)
        center = pts.mean(axis=0)
        dx = center[0] - cx
        dy = center[1] - cy
        d2 = dx * dx + dy * dy
        if d2 < best_dist:
            best_dist = d2
            best_id = int(mid)
    return best_id

def build_output_name(
    section_id: str,
    corner_ids: Dict[str, int],
    canvas: Tuple[int, int],
    dict_name: str,
    ext: str
) -> str:
    cid = f"TL{corner_ids['TL']}_TR{corner_ids['TR']}_BR{corner_ids['BR']}_BL{corner_ids['BL']}"
    w, h = canvas
    return f"{section_id}__ids={cid}__{w}x{h}__dict={dict_name}{ext}"
