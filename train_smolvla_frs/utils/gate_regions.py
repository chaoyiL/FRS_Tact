"""Shared three-region gate policy and six-bin evaluation layout."""

from __future__ import annotations

DEFAULT_RANK_LOW_GATE_THRESHOLD = 0.3
DEFAULT_RANK_HIGH_GATE_THRESHOLD = 0.7

GATE_BIN_SPECS: tuple[tuple[str, float, float], ...] = (
    ("00_01", 0.0, 0.1),
    ("01_03", 0.1, 0.3),
    ("03_05", 0.3, 0.5),
    ("05_07", 0.5, 0.7),
    ("07_09", 0.7, 0.9),
    ("09_10", 0.9, 1.0),
)
