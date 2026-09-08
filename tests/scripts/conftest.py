"""Make `scripts/` importable by test modules under `tests/scripts/`.

`scripts/` holds standalone CLI scripts, not a package inside `src/`, so it
isn't on `sys.path` by default under pytest's import mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
