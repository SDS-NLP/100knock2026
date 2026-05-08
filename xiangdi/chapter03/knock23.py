# 23. セクション構造
# 記事中に含まれるセクション名とそのレベル（例えば”== セクション名 ==”なら1）を表示せよ。

import re

with open("/Users/caitlyn/Documents/uk.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"^(=+)\s*(.*?)\s*\1$"

# ^: Match start of line.
# (=+): Capture a sequence of = characters at the start of a line.
# \s*: Match 0 or more whitespace characters
# (.*?): Capture any content, but as little as possible.
# \1: Match the exact same content as captured earlier (=+).
# $: Match end of line.

for line in text.split("\n"):
    match = re.match(pattern, line)
    if match:
        level = len(match.group(1)) - 1
        section_name = match.group(2)
        print(f"Section Name: {section_name}; Level:{level}")