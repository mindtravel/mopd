"""Launcher for the checkpoint benchmark service (mopd.serving.service; docs/evaluation/bench-service.md).

  python scripts/bench_service.py [--port 18090]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mopd.serving.service import main

if __name__ == "__main__":
    main()
