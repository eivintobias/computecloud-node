"""Pytest configuration for the standalone node_client package.

Ensures the local ``src/`` directory is on ``sys.path`` ahead of any
installed ``computecloud_node`` package, so tests run against the in-tree
source (which may be newer than the published package).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# If computecloud_node was already imported from a different location (e.g.
# an installed copy from the separate computecloud-node repo), evict it from
# the module cache so subsequent imports resolve to the local src/ version.
for _mod_name in list(sys.modules):
    if _mod_name == "computecloud_node" or _mod_name.startswith("computecloud_node."):
        _mod = sys.modules[_mod_name]
        _mod_file = getattr(_mod, "__file__", "") or ""
        if _mod_file and not _mod_file.startswith(str(_SRC)):
            del sys.modules[_mod_name]

