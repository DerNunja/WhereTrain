import os
import re
import numpy as np
from typing import List, Optional, Dict, Any, Tuple


Point = Tuple[int, int]

def parse_section_from_filename(path: str) -> Dict[str, Any]:
    base = os.path.basename(path)
    m = re.match(r"(?P<section>[^_]+(?:_[^_]+)*)__ids=(?P<ids>[^_]+)__(?P<w>\d+)x(?P<h>\d+)", base)
    if m:
        section_id = m.group("section")
        ids_str = m.group("ids")
        w, h = int(m.group("w")), int(m.group("h"))

        corner_ids = {}
        for part in ids_str.split("_"):
            mm = re.match(r"(TL|TR|BR|BL)(\d+)", part)
            if mm:
                corner_ids[mm.group(1)] = int(mm.group(2))

        return {"section_id": section_id, "corner_ids": corner_ids or None, "canvas": (w, h), "raw": base}

    section_id = os.path.splitext(base)[0]
    return {"section_id": section_id, "corner_ids": None, "canvas": None, "raw": base}


def default_output_json_path(image_path: str) -> str:
    folder = os.path.dirname(image_path)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(folder if folder else ".", f"{stem}__trackmap.json")

def to_py_points(pts: List[Point]) -> List[List[int]]:
    """
    Converts [(x,y), ...] where x/y may be numpy scalars into JSON-safe lists [[x,y],...]
    """
    out: List[List[int]] = []
    for p in pts:
        out.append([int(p[0]), int(p[1])])
    return out

def track_sort_key(tid: str):
    if len(tid) >= 2 and tid[0].isalpha() and tid[1:].isdigit():
        return (tid[0], int(tid[1:]), tid)
    return ("~", 999999, tid)


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def polyline_to_band(points: List[Point], width_px: float) -> List[Point]:
    if len(points) < 2:
        return []

    pts = np.array(points, dtype=np.float32)
    N = len(pts)
    half = float(width_px) / 2.0

    seg_dirs = pts[1:] - pts[:-1]
    seg_dirs = np.array([_normalize(d) for d in seg_dirs], dtype=np.float32)
    seg_normals = np.stack([-seg_dirs[:, 1], seg_dirs[:, 0]], axis=1)

    v_normals = np.zeros((N, 2), dtype=np.float32)
    v_normals[0] = seg_normals[0]
    v_normals[-1] = seg_normals[-1]

    for i in range(1, N - 1):
        n = seg_normals[i - 1] + seg_normals[i]
        n = _normalize(n)
        if np.linalg.norm(n) < 1e-6:
            n = seg_normals[i]
        v_normals[i] = n

    left = pts + v_normals * half
    right = pts - v_normals * half

    poly = np.vstack([left, right[::-1]])
    poly = np.round(poly).astype(np.int32)

    cleaned = [tuple(poly[0])]
    for p in poly[1:]:
        t = tuple(p)
        if t != cleaned[-1]:
            cleaned.append(t)

    return cleaned if len(cleaned) >= 3 else []


def next_free_track_id(existing: set, start_from: int = 1) -> str:
    i = max(1, start_from)
    while True:
        tid = f"G{i}"
        if tid not in existing:
            return tid
        i += 1


def parse_track_number(track_id: str) -> Optional[int]:
    if len(track_id) >= 2 and track_id[0].upper() == "G" and track_id[1:].isdigit():
        return int(track_id[1:])
    return None