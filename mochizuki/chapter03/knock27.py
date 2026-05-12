import re
import knock25

def remove_emphasis(text):
    return re.sub(r"'{2,5}", '', text)

def remove_links(text):
    return re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)

for k, v in knock25.info.items():
    print(f'{k}: {remove_links(remove_emphasis(v))}')
