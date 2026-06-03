import re
import math
from collections import Counter
import MeCab

text = './chapter04/kokoro.txt'

def clean_text(text):
    text = re.sub(r'《.*?》', '', text)
    text = re.sub(r'｜', '', text)

    return text

sections = []
current_section_lines = []

with open(text, 'r', encoding='utf-8') as f:
    for line in f:
        line_str = line.strip()
        