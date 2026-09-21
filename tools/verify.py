"""一键门禁：lint -> type-check -> unit/invariant tests -> P0 emoji scan。

用法（仓库根目录）：
    uv run python tools/verify.py        # 或 .venv/Scripts/python tools/verify.py
非零退出码 = 门禁失败。CI 与交付前必跑。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS: list[tuple[str, list[str]]] = [
    ("ruff lint", [str(ROOT / ".venv" / "Scripts" / "ruff.exe"), "check", "src", "tests"]),
    ("mypy type-check", [str(ROOT / ".venv" / "Scripts" / "mypy.exe"), "src"]),
    ("pytest", [PY, "-m", "pytest", "tests", "--basetemp=data/.pytest-tmp",
                "-p", "no:cacheprovider", "-q"]),
    ("emoji scan", [PY, "tools/scan_emoji.py"]),
]


def main() -> int:
    failures: list[str] = []
    for name, cmd in STEPS:
        print(f"\n=== {name} ===")
        cmd = [c for c in cmd if Path(str(c)).exists() or not str(c).endswith(".exe")] \
            if name != "pytest" else cmd
        try:
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=300)
            out = (proc.stdout or "") + (proc.stderr or "")
            print(out.strip()[-2000:])
            if proc.returncode != 0:
                failures.append(f"{name} (exit {proc.returncode})")
            else:
                print(f"--- {name}: OK")
        except FileNotFoundError as exc:
            failures.append(f"{name} ({exc})")
    print("\n" + "=" * 50)
    if failures:
        print("VERIFY FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERIFY PASSED: lint + types + tests + emoji scan all green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
