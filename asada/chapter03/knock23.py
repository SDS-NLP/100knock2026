import json
import re
from pathlib import Path

text = str(json.load(Path("uk.json").open(encoding="UTF-8")))
print(
    "\n".join(
        f"level {len(section.group(1))}: {section.group(2)[1:-1].strip()}"
        for section in re.finditer(r"(=+)(=\s*.+?\s*=)=+", text)
    )
)
