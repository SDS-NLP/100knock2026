import re
from knock20 import extract_uk_text

UK_TEXT = extract_uk_text()

def extract_infobox(text):

    pattern = re.compile("\|(.+?)\s=\s*(.+)")
    ans     = {}
    
    for line in text.split('\n'):
        r = re.search(pattern, line)
        if r:
            ans[r[1]] = r[2]
    return ans

if __name__ == "__main__":
    infobox = extract_infobox(UK_TEXT)

    for key, value in infobox.items():
        print(f"{key}: {value}")