import math

def split_file(N):
    """ファイルを行単位でN分割し、別のファイルに格納せよ。
    """
    with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
        lines = f.readlines()
        num = len(lines)
        line_num = math.ceil(num / N)

    for i in range(0, num, line_num):
        content = lines[i:i+line_num]
        file_index = i // line_num + 1

        with open(f"file_{file_index}.txt", "w") as out:
            out.writelines(content)

split_file(10)

# split -l 278 /Users/caitlyn/Downloads/popular-names.txt