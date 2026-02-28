from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple


DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"
DIRECTION_STILL = "still"


@dataclass
class TrackState:
    positions: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    smoothed_positions: Deque[float] = field(default_factory=lambda: deque(maxlen=2))
    direction: str = DIRECTION_STILL

    def update(self, s_norm_raw: float, direction_threshold: float = 0.005) -> Tuple[float, str]:
        self.positions.append(float(s_norm_raw))
        s_norm_smooth = sum(self.positions) / len(self.positions)

        self.smoothed_positions.append(s_norm_smooth)
        if len(self.smoothed_positions) >= 2:
            delta = self.smoothed_positions[-1] - self.smoothed_positions[-2]
            if delta > direction_threshold:
                self.direction = DIRECTION_FORWARD
            elif delta < -direction_threshold:
                self.direction = DIRECTION_BACKWARD
            else:
                self.direction = DIRECTION_STILL

        return s_norm_smooth, self.direction


def make_track_key(section_id: str, track_id: str) -> str:
    return f"{section_id}__{track_id}"


def get_or_create_state(track_states: Dict[str, TrackState], section_id: str, track_id: str) -> TrackState:
    key = make_track_key(section_id, track_id)
    state = track_states.get(key)
    if state is None:
        state = TrackState()
        track_states[key] = state
    return state
