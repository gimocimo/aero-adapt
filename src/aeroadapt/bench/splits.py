"""Airframe splits (M2 spec §4.2): 80 / 10 / 10 by a hash of the airframe id, so an airframe's
split never depends on which fleet or sample it appears in."""

from __future__ import annotations

import hashlib
from typing import Literal

Split = Literal["train", "val", "test"]


def split_of(airframe_id: str) -> Split:
    h = int.from_bytes(hashlib.sha256(airframe_id.encode()).digest()[:8], "big") % 10
    return "test" if h == 9 else "val" if h == 8 else "train"
