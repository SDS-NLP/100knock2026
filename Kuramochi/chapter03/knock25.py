import re
from knock20 import extract_uk_text

UK_text = extract_uk_text()

pattern = r'{{基礎情報(.*?)}}'
result = re.findall(pattern, UK_text, re.DOTALL)
if result:
    result[0] += "\n"
    print(result[0])
else:
    print("基礎情報が見つかりませんでした。")

# pattern = r'(?<=\\\\n\|)(.*?) *= *(.*?)(?=\\\\n)'
# result2 = re.findall(pattern, result[0])
# inf_dic = {}
# for i, j in result2:
#   inf_dic[i] = j
# inf_dic
