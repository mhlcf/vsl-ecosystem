"""Root conftest for Part 3.

Registers part3's ``backend`` / ``frontend`` packages under unique aliases
(``p3backend``, ``p3frontend``) so both VSL parts can be tested in a single
pytest process without package collisions or path-shadowing surprises.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_PART3 = Path(__file__).resolve().parent

# unique aliases (immune to any pytest sys.path juggling)
for _alias, _name in (("p3backend", "backend"),
                      ("p3frontend", "frontend")):
    if _alias not in sys.modules:
        _pkg = types.ModuleType(_alias)
        _pkg.__path__ = [str(_PART3 / _name)]
        sys.modules[_alias] = _pkg

# part3's source also uses plain `backend` / `frontend` imports; point those
# names at part3's own packages so they work regardless of sys.path order
for _name in ("backend", "frontend"):
    if _name not in sys.modules:
        _pkg = types.ModuleType(_name)
        _pkg.__path__ = [str(_PART3 / _name)]
        sys.modules[_name] = _pkg