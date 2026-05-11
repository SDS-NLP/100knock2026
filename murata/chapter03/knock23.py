with open('uk.txt', encoding='utf-8') as f:
    uk_text = f.read()
   
import re 
pattern = re.compile(r"^(=+)\s*(.+?)\s*\1$", re.MULTILINE)
for m in pattern.finditer(uk_text):
    level = len(m.group(1)) - 1
    name = m.group(2)
    print(f"{name} (level {level})")
