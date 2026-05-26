import re
from knock00 import uk_txt

# 基礎情報テンプレート全体を取得
pattern = r"\{\{基礎情報.*?\n(.*?)\n\}\}"

match = re.search(pattern, uk_txt, re.DOTALL)

basic_info = {}

if match:
    template_text = match.group(1)

    field_pattern = r"\|(.+?)\s*=\s*(.+?)(?=\n\||\n$)"

    fields = re.findall(field_pattern, template_text, re.DOTALL)

    for key, value in fields:
        basic_info[key] = value.strip()

if __name__ == "__main__":
    print(basic_info)
