"""
Classical color-segmentation vision for the warehouse pickup task.

The four task objects are colored boxes (red / blue / green / yellow). This
module takes an RGB image from the Husky's head camera, thresholds pixels by
color channel to produce a mask per color, and returns each color's pixel
count plus centroid. Combined with the language parser in
`instruction_parser.py`, this grounds a natural-language instruction like
"pick up the red box" to a specific object in the scene.

Deliberately simple — no neural perception stack — but it is a real
vision-in-the-loop module: the pipeline actually has to look at pixels from
the camera to confirm where the target is.
"""
from typing import Dict, Optional, Tuple

import numpy as np


MIN_PIXELS = 20  # reject tiny blobs from anti-aliasing / shadows


def _mask_for(rgb: np.ndarray, color: str) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    # Thresholds empirically tuned to the rendered scene. PyBullet's default
    # lighting compresses pure box colors significantly: e.g. a pure-red box
    # renders with R ~80-119, and a pure-blue box with B ~180-224. So
    # thresholds use dominant-channel checks plus channel separation.
    if color == "red":
        return (r > 70) & (r > g + 20) & (r > b + 25)
    if color == "blue":
        return (b > 150) & (b > r + 35) & (b > g + 25)
    if color == "green":
        return (g > 100) & (g > r + 25) & (g > b + 25)
    if color == "yellow":
        return (r > 150) & (g > 150) & (r > b + 30) & (g > b + 30)
    return np.zeros(rgb.shape[:2], dtype=bool)


def detect_color(rgb: np.ndarray, color: str
                  ) -> Tuple[bool, Optional[Tuple[float, float]], int]:
    """Detect a single colored blob. Returns (found, (cx, cy) in pixel space, pixel_count)."""
    mask = _mask_for(rgb, color)
    count = int(mask.sum())
    if count < MIN_PIXELS:
        return False, None, count
    ys, xs = np.where(mask)
    return True, (float(xs.mean()), float(ys.mean())), count


def detect_all(rgb: np.ndarray) -> Dict[str, Dict]:
    """Detect every supported color. Returns color -> {centroid, pixel_count}
    for those found."""
    out: Dict[str, Dict] = {}
    for color in ("red", "blue", "green", "yellow"):
        found, centroid, count = detect_color(rgb, color)
        if found:
            out[color] = {"centroid": centroid, "pixel_count": count}
    return out
