import re
from knock06 import cleaned_info

cleaned2_info = {}

for key, value in cleaned_info.items():
    value = re.sub(r"\[\[(?:[^|\]]*\|)?([^|\]]+)\]\]", r"\1", value)

    cleaned2_info[key] = value

if __name__ == "__main__":
    print(cleaned2_info)
