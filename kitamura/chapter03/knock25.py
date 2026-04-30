import re
file = "uk_article.txt"

with open(file, "r", encoding="utf-8") as f:
    content = f.read()

pattern1 = r"{{基礎情報 国\n(.*)\n}}"
match = re.search(pattern1, content, re.DOTALL) 
# findall：リストを返す search：最初の一つをmatchオブジェクトで返す
# re.DOTALL：複数行の抜き出し
if match:
    basic_info = match.group(1)

pattern2 = r"^\|(.*?)\s*=\s*(.*?)(?=\n^\||\n^}|$)"
field_names = re.findall(pattern2, basic_info, re.DOTALL | re.MULTILINE)
# pattern2 = r"\|(.*?)\s*=\s*(.*)"
# field_names = re.findall(pattern2, basic_info, re.MULTILINE)

result = {}
for field_name, v in field_names:
    result[field_name] = v


for key, value in result.items():
    print(f"{key}:{value}")

print(result["国旗画像"])