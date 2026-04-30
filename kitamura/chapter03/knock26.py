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

final_dict = {}

for key, value in result.items():
    new_value = re.sub(pattern_emp, "", value)
    final_dict[key] = new_value

print(final_dict)