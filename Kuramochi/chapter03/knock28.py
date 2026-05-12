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
    
def remove_mk(v):
    r1 = re.compile("'+")
    r2 = re.compile("\[\[(.+\||)(.+?)\]\]")
    r3 = re.compile("\{\{(.+\||)(.+?)\}\}")
    r4 = re.compile("<\s*?/*?\s*?br\s*?/*?\s*>")
    v = r1.sub("", v)
    v = r2.sub(r"\2", v)
    v = r3.sub(r"\2", v)
    v = r4.sub("", v)
    return v


r   = re.compile("\[\[(.+\||)(.+?)\]\]")
ans = {k: r.sub(r"\2", remove_mk(v)) for k, v in infobox.items()}
print(remove_inner_links(remove_stress(ans)))

