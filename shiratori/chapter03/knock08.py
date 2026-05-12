import re
from knock07 import cleaned2_info

cleaned3_info = {}

for key, value in cleaned2_info.items():
    # 外部リンク
    value = re.sub(r"\[https?://[^\s]+ ([^\]]+)\]", r"\1", value)
    value = re.sub(r"\[(https?://[^\]]+)\]", r"\1", value)

    # HTMLタグ
    value = re.sub(r"<ref.*?>.*?</ref>", "", value)
    value = re.sub(r"<ref.*?/>", "", value)
    value = re.sub(r"<br\s*/?>", "", value)

    # テンプレート
    value = re.sub(r"\{\{lang\|[^|]+\|([^}]+)\}\}", r"\1", value)

    cleaned3_info[key] = value.strip()

if __name__ == "__main__":
    print(cleaned3_info)
