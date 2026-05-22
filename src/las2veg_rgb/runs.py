"""Run directory helpers.

Each Phase 1 invocation creates a timestamped subdirectory under the user-specified
output root (e.g. data/output/run_20260522_063000/). A 'latest' junction (Windows)
or symlink (POSIX) in the output root always points to the most recent run, so
Phase 2 / Phase 3 can default to `<root>/latest/...` for convenience.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

RUN_PREFIX = "run_"
LATEST_NAME = "latest"


def create_run_dir(output_root: Path, suffix: str | None = None) -> Path:
    """Create a new timestamped run directory under output_root.

    suffix is appended after the timestamp (e.g. mesh size: '1m', '2.5m').
    Returns the absolute path of the new run directory.
    """
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{RUN_PREFIX}{stamp}"
    if suffix:
        base_name = f"{base_name}_{suffix}"
    run_dir = output_root / base_name
    # Avoid collision if invoked twice within the same second
    collision = 0
    while run_dir.exists():
        collision += 1
        run_dir = output_root / f"{base_name}_{collision}"
    run_dir.mkdir(parents=True)
    return run_dir


def update_latest_link(output_root: Path, run_dir: Path) -> None:
    """Point output_root/latest at run_dir.

    On Windows uses a directory junction (no admin required, works across drives).
    On POSIX uses a symlink.
    """
    output_root = output_root.resolve()
    run_dir = run_dir.resolve()
    latest = output_root / LATEST_NAME

    # Remove existing latest link/dir if any
    if latest.exists() or latest.is_symlink():
        try:
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            else:
                # On Windows a junction looks like a directory; rmdir works.
                latest.rmdir()
        except OSError:
            # Last resort: try removing as junction via cmd
            subprocess.run(
                ["cmd", "/c", "rmdir", str(latest)],
                check=False, capture_output=True,
            )

    if os.name == "nt":
        # mklink /J <link> <target>
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(latest), str(run_dir)],
            check=True, capture_output=True,
        )
    else:
        latest.symlink_to(run_dir, target_is_directory=True)


def find_latest_run(output_root: Path) -> Path | None:
    """Return the most recent run directory under output_root, or None.

    Prefers the 'latest' link if it points to a valid directory; otherwise
    scans for run_* directories and picks the newest by name.
    """
    output_root = output_root.resolve()
    if not output_root.is_dir():
        return None

    latest = output_root / LATEST_NAME
    if latest.is_dir():
        return latest.resolve()

    candidates = sorted(
        [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith(RUN_PREFIX)],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None
