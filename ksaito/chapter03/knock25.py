import re


def extract_basic_info(input_file):
    with open(input_file, encoding="utf-8") as f:
        text = f.read()

    m = re.search(r"\{\{基礎情報.*?\n(.*?)\n\}\}", text, re.DOTALL)
    template_body = m.group(1)

    pattern = r"^\|(.+?)\s*=\s*(.*?)(?=\n\||\n\}\}$|\Z)"
    fields = {}
    for m in re.finditer(pattern, template_body, re.MULTILINE | re.DOTALL):
        name = m.group(1).strip()
        value = m.group(2).strip()
        fields[name] = value

    return fields


if __name__ == "__main__":
    input_file = "data/uk.txt"
    info = extract_basic_info(input_file)
    for k, v in info.items():
        print(f"{k}: {v}")
