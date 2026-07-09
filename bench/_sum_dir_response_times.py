import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
key = sys.argv[2] if len(sys.argv) > 2 else "response_time"
tot = 0.0
n = 0
for p in sorted(d.glob("*.jsonl")):
    txt = p.read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        continue
    # single-line jsonl preferred
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            nums = re.findall(rf'"{re.escape(key)}"\s*:\s*([0-9.eE+-]+)', line)
            if nums:
                tot += float(nums[0])
                n += 1
            continue
        if isinstance(o, dict) and key in o and isinstance(o[key], (int, float)):
            tot += float(o[key])
            n += 1
print(d, "files", len(list(d.glob('*.jsonl'))), "samples", n, "sum_s", tot)
