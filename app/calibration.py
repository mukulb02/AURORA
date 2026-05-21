from collections import deque
from typing import Tuple
import numpy as np

from app.config import (
    INITIAL_K,
    ROLLING_WINDOW,
    CONF_MARGIN_DEFAULT
)

class DepthManager:

    def __init__(self):
        self.hist = {
            b: deque(maxlen=ROLLING_WINDOW)
            for b in ["S", "M", "L", "XL"]
        }

    def suggest(self, bucket: str) -> Tuple[int, int]:

        kS, kD = INITIAL_K[bucket]

        used = list(self.hist[bucket])

        if used:
            medS = int(np.median([u[0] for u in used]))
            medD = int(np.median([u[1] for u in used]))

            kS = max(5, int(0.9 * kS + 0.1 * (medS + 2)))
            kD = max(5, int(0.9 * kD + 0.1 * (medD + 2)))

        return kS, kD

    def record(self, bucket: str, kS: int, kD: int):
        self.hist[bucket].append((kS, kD))


class Calibrator:

    def __init__(self, default_delta=CONF_MARGIN_DEFAULT):
        self.default = default_delta
        self.margins = deque(maxlen=ROLLING_WINDOW)

    def delta(self) -> float:

        if not self.margins:
            return self.default

        return float(np.percentile(self.margins, 40))

    def record(self, margin: float):
        self.margins.append(margin)


depth_manager = DepthManager()
calibrator = Calibrator()