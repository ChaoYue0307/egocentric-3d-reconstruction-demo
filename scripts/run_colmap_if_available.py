#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generated COLMAP commands when COLMAP is available, otherwise write a clear readiness report.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sample_demo"))
    parser.add_argument("--commands", type=Path, default=Path("outputs/sample_demo/colmap_commands.sh"))
    parser.add_argument("--run", action="store_true", help="Actually execute COLMAP commands. Without this flag, only a readiness report is written.")
    return parser.parse_args()


def write_report(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "colmap_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    status = "available" if report["colmap_available"] else "not available"
    text = f"""# COLMAP Run Report

- COLMAP status: {status}
- Command template: `{report["commands"]}`
- Executed: {report["executed"]}
- Exit code: {report["exit_code"]}

## Notes

Install COLMAP and rerun:

```bash
python scripts/run_colmap_if_available.py --run
```
"""
    (output_dir / "colmap_run_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    colmap_path = shutil.which("colmap")
    report = {
        "colmap_available": colmap_path is not None,
        "colmap_path": colmap_path,
        "commands": str(args.commands),
        "executed": False,
        "exit_code": None,
    }
    if colmap_path and args.run:
        result = subprocess.run(["bash", str(args.commands)], check=False)
        report["executed"] = True
        report["exit_code"] = result.returncode
    write_report(args.output_dir, report)
    print(json.dumps(report))
    return int(report["exit_code"] or 0)


if __name__ == "__main__":
    raise SystemExit(main())
