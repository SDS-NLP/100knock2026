with open('output_knock20.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# "[[Category:" で始まる行を抽出
category_lines = [line.strip() for line in lines if line.startswith('[[Category:')]

for line in category_lines:
    print(line)