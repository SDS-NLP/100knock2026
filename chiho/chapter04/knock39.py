from collections import Counter
from math import log10
from pathlib import Path
import re

import MeCab


def clean_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = re.sub(r"《[^》]+》", "", text)
    text = re.sub(r"［＃.*?］", "", text)
    text = re.sub(r"^.+?｜", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[一二三四五六七八九十百]+$", "", text, flags=re.MULTILINE)
    return text


file_path = Path(__file__).with_name("kokoro.txt")
with open(file_path, "r", encoding="utf-8") as f:
    text = clean_text(f.read())

mecab = MeCab.Tagger(r"-r /opt/homebrew/etc/mecabrc -d /opt/homebrew/lib/mecab/dic/ipadic")
node = mecab.parseToNode(text)

word_counter = Counter()

while node:
    if node.surface:
        features = node.feature.split(",")
        if features[0] not in {"BOS/EOS", "記号"}:
            word_counter[node.surface] += 1
    node = node.next

frequencies = [count for _, count in word_counter.most_common()]
ranks = list(range(1, len(frequencies) + 1))

width = 900
height = 600
margin_left = 80
margin_right = 40
margin_top = 60
margin_bottom = 80
plot_width = width - margin_left - margin_right
plot_height = height - margin_top - margin_bottom

max_rank_log = log10(max(ranks))
max_freq_log = log10(max(frequencies))

points = []
for rank, frequency in zip(ranks, frequencies):
    x = margin_left + (log10(rank) / max_rank_log) * plot_width
    y = margin_top + plot_height - (log10(frequency) / max_freq_log) * plot_height
    points.append(f"{x:.2f},{y:.2f}")

svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white" />
  <text x="{width / 2}" y="30" text-anchor="middle" font-size="20">Word Frequency Rank Plot (log-log)</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="black" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="black" />
  <polyline fill="none" stroke="#1f77b4" stroke-width="1.5" points="{' '.join(points)}" />
  <text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="16">Rank</text>
  <text x="25" y="{height / 2}" text-anchor="middle" font-size="16" transform="rotate(-90 25 {height / 2})">Frequency</text>
</svg>
"""

output_path = Path(__file__).with_name("knock39.svg")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(svg_content)

print(output_path)
