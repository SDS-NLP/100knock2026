import re
from knock05 import basic_info

cleaned_info = {}

for key, value in basic_info.items():
    cleaned_value = re.sub(r"'{2,5}", "", value)
    cleaned_info[key] = cleaned_value

if __name__ == "__main__":
    print(cleaned_info)
