import re
from knock20 import extract_uk_text
from knock25 import extract_infobox

UK_TEXT  = extract_uk_text()
UK_LINES = UK_TEXT.split('\n')
infobox  = extract_infobox(UK_TEXT)

def remove_stress(dc):
    r = re.compile("'+")
    return {k: r.sub("", v) for k, v in dc.items()}

def remove_inner_links(dc):
    r = re.compile("\[\[(.+\||)(.+?)\]\]")
    return {k: r.sub(r"\2", v) for k, v in dc.items()}

print(remove_inner_links(remove_stress(infobox)))