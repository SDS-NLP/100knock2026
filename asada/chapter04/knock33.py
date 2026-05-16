import CaboCha

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

pumpkin = CaboCha.Parser()
result = (
    pumpkin.parse(text).toString(CaboCha.FORMAT_LATTICE).split("\n")[:-2]
)  # 最後尾の"EOS"と""を除外

data = {}
current_chunk = None
for line in result:
    if line[0] == "*":
        current_chunk = tuple(line.split()[1:3])
        data[current_chunk] = []
    else:
        data[current_chunk].append(line.split("\t")[0])
print(data)

dependencies = ""
for key in data.keys():
    target_id = key[1][:-1]
    dependencies += (
        f"{data[key]}\t{data[next(k for k in data.keys() if k[0] == target_id)]}\n"
        if target_id != "-1"
        else f"{data[key]}\t\n"
    )
print(dependencies)
