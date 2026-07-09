import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2] if len(sys.argv) > 2 else "time_elapsed"
text = path.read_text(encoding="utf-8", errors="replace")
nums = [float(x) for x in re.findall(rf'"{re.escape(key)}"\s*:\s*([0-9.eE+-]+)', text)]
print(path.name, len(nums), sum(nums))
