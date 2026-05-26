import re

template_pattern = r"^\{\{基礎情報.*?$(.*?)^\}\}"

with open("イギリス.txt", "r") as fp:
    template_match = re.search(template_pattern, fp.read(), re.DOTALL | re.MULTILINE)

infobox_dict = {}

if template_match:
    infobox_content = template_match.group(1)

    field_pattern = r"^\|(.+?)\s*=\s*(.+?)(?=\n\||^$|^\}\})"
    fields = re.findall(field_pattern, infobox_content, re.MULTILINE | re.DOTALL)

    for key, value in fields:
        infobox_dict[key.strip()] = value.strip()


def remove_markup(text):
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{(?:[^|}]+\|)+([^|}]+)\}\}", r"\1", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[http[^\s]*\s?([^\]]*)]", r"\1", text)

    return text.strip()


cleaned_dict = {k: remove_markup(v) for k, v in infobox_dict.items()}

for key, value in cleaned_dict.items():
    print(f"{key}: {value}")
