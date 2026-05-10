import json
import re
from pathlib import Path

text = str(json.load(Path("uk.json").open(encoding="utf-8")))
raw_info = re.sub(
    r"'{2,5}",
    "",
    re.search(r"\{\{基礎情報.*?\|(.*?)\}\}(?:\n)*'", text, re.DOTALL).group(1),
)

raw_info = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", raw_info)
raw_info = re.sub(r"\[\[(.*?)\]\]", r"\1", raw_info)
raw_info = re.sub(r"\*", "", raw_info)
raw_info = re.sub(r"\</*?ref(:?erences/)*\>", "", raw_info)
raw_info = re.sub(r"\<br\s*?/\>", "", raw_info)
raw_info = re.sub(r"/\>", "", raw_info)

basic_info = re.split(r"\n\|", raw_info)
print({re.split(r"=", i)[0].strip(): re.split(r"=", i)[-1].strip() for i in basic_info})
