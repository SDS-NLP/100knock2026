import re

with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    match = re.match(r'(={2,})\s*(.+?)\s*\1', line.strip())
    if match:
        level = len(match.group(1)) - 1  
        section_name = match.group(2)
        print(f'レベル{level}: {section_name}')