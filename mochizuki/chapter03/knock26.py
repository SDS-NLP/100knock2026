import re
import knock25

def remove_emphasis(text):
    return re.sub(r"'{2,5}", '', text)

for k, v in knock25.info.items():
    print(f'{k}: {remove_emphasis(v)}')
