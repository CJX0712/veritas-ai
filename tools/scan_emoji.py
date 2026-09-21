"""P0 门禁：全仓扫描 emoji 表情（禁止 emoji 作为功能图标）。

命中数 > 0 时以非零码退出。允许出现在 docs/*.md 的 UGC 示例中（本项目无此场景）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF"
    "\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
    "\U0001F100-\U0001F64F\U0000200D\U000020E3]"
)

SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", "data", ".pytest-tmp"}
SKIP_FILES = {"_probe1.txt", "uv.lock"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES or path.suffix not in {
            ".py", ".html", ".md", ".toml", ".yml", ".yaml", ".json",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if EMOJI_RE.search(line):
                hits.append(f"{path.relative_to(root)}:{lineno}")
    if hits:
        print(f"EMOJI FOUND ({len(hits)}):")
        for h in hits:
            print(f"  {h}")
        return 1
    print("emoji scan: clean (0 hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
