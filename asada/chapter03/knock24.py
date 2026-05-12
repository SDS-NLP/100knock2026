import json
import re
from pathlib import Path

text = str(json.load(Path("uk.json").open(encoding="UTF-8")))
print("\n".join(re.findall(r"\[\[(?:File|ファイル):(.*?)(?:\||\])", text)))
