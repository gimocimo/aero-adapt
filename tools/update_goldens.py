"""Regenerate the golden-flight metrics (tests/regression/goldens.json).

Run this only for an intended behaviour change, and say why in the commit message.

Usage: ``uv run python tools/update_goldens.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from aeroadapt.scenarios.goldens import GOLDEN_SCENARIOS, metrics

OUT = Path(__file__).resolve().parents[1] / "tests" / "regression" / "goldens.json"


def main() -> None:
    goldens = {sid: metrics(sid) for sid in GOLDEN_SCENARIOS}
    OUT.write_text(json.dumps(goldens, indent=2, sort_keys=True) + "\n")
    for sid, m in goldens.items():
        print(sid, m)


if __name__ == "__main__":
    main()
