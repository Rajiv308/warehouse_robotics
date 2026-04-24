"""
Language-grounding for task instructions.

A deterministic, keyword-based parser that extracts the target object color
from a natural-language instruction string. Deliberately lightweight — no
neural LM involved — but lets the rest of the pipeline treat the instruction
as the load-bearing input instead of a cosmetic field.
"""
import re
from typing import Optional

COLOR_TO_IDX = {
    "red": 0,
    "blue": 1,
    "green": 2,
    "yellow": 3,
}

IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}


def parse_target_color(instruction: str) -> Optional[str]:
    """Return the first color keyword found in the instruction, or None."""
    if not instruction:
        return None
    text = instruction.lower()
    for color in COLOR_TO_IDX:
        if re.search(rf"\b{color}\b", text):
            return color
    return None


def color_to_idx(color: Optional[str]) -> Optional[int]:
    if color is None:
        return None
    return COLOR_TO_IDX.get(color)


def idx_to_color(idx: int) -> Optional[str]:
    return IDX_TO_COLOR.get(idx)
