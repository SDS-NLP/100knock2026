import re
file = "uk_article.txt"

with open(file, "r", encoding="utf-8") as f:
    content = f.read()

pattern1 = r"{{基礎情報 国\n(.*)\n}}"
match = re.search(pattern1, content, re.DOTALL) 

if match:
    basic_info = match.group(1)

pattern2 = r"^\|(.*?)\s*=\s*(.*?)(?=\n^\||\n^}|$)"
field_names = re.findall(pattern2, basic_info, re.DOTALL | re.MULTILINE)

result = {}
for field_name, v in field_names:
    result[field_name] = v

pattern_emp = r"'{2,5}"
pattern_link = r"\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]"

dict_27 = {}

for key, value in result.items():
    without_emp_value = re.sub(pattern_emp, "", value)
    without_link_value = re.sub(pattern_link, r"\1", without_emp_value)
    dict_27[key] = without_link_value

for key, val in dict_27.items():
    print(f"{key}:{val}")
