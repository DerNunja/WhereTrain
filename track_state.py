from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple


DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"
DIRECTION_STILL = "still"

DIRECTION_THRESHOLD = 0.003

SMOOTH_WINDOW = 5

DIRECTION_WINDOW = 8


@dataclass
class TrackState:
    positions: Deque[float] = field(default_factory=lambda: deque(maxlen=SMOOTH_WINDOW))
    smoothed_history: Deque[float] = field(default_factory=lambda: deque(maxlen=DIRECTION_WINDOW))
    direction: str = DIRECTION_STILL

    def update(self, s_norm_raw: float) -> Tuple[float, str]:
        self.positions.append(float(s_norm_raw))
        s_norm_smooth = sum(self.positions) / len(self.positions)
        self.smoothed_history.append(s_norm_smooth)

        if len(self.smoothed_history) >= 3:
            delta = self.smoothed_history[-1] - self.smoothed_history[0]
            if delta > DIRECTION_THRESHOLD:
                self.direction = DIRECTION_FORWARD
            elif delta < -DIRECTION_THRESHOLD:
                self.direction = DIRECTION_BACKWARD
            else:
                self.direction = DIRECTION_STILL

        return s_norm_smooth, self.direction


def make_track_key(section_id: str, track_id: str) -> str:
    return f"{section_id}__{track_id}"


def get_or_create_state(
    track_states: Dict[str, "TrackState"], section_id: str, track_id: str
) -> "TrackState":
    key = make_track_key(section_id, track_id)
    state = track_states.get(key)
    if state is None:
        state = TrackState()
        track_states[key] = state
    return state