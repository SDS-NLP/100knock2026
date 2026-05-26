import json
import re
from pathlib import Path

text = str(json.load(Path("uk.json").open(encoding="UTF-8")))
basic_info = re.split(
    r"\n\|",
    re.search(r"\{\{基礎情報.*?\|(.*?)\}\}(:?\n)*'", text, re.DOTALL).group(1),
)

print({re.split(r"=", i)[0].strip(): re.split(r"=", i)[-1].strip() for i in basic_info})
