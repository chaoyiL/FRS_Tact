#!/usr/bin/env python
"""Thin CLI wrapper for package-owned tactile embedding precomputation."""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_vtsmolvla.precompute import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
