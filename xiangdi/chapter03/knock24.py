# 24. ファイル参照の抽出
# 記事から参照されているメディアファイルをすべて抜き出せ。

import re

with open("/Users/caitlyn/Documents/uk.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"\[\[ファイル:(.*?)(?:\|.*)?\]\]"

# (.*?): Capture any content, but as little as possible.
# (?: ... ): （non-capturing group)
# An optional | followed by any content (not captured).

files = re.findall(pattern, text)

with open("file.txt", "w", encoding="utf-8") as f:
    f.writelines(file + "\n" for file in files)