"""Root conftest for Part 2.

Registers part2's ``backend`` / ``avatar`` / ``frontend`` packages under
unique aliases (``p2backend``, ``p2avatar``, ``p2frontend``) so both VSL
parts can be tested in a single pytest process without package collisions.
Runs before any Part 2 test module and for every subdirectory.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

_PART2 = Path(__file__).resolve().parent

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

for _alias, _name in (("p2backend", "backend"),
                      ("p2avatar", "avatar"),
                      ("p2frontend", "frontend")):
    if _alias not in sys.modules:
        _pkg = types.ModuleType(_alias)
        _pkg.__path__ = [str(_PART2 / _name)]
        sys.modules[_alias] = _pkg