import subprocess

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

result = subprocess.run(
    ["cabocha", "-f1"],
    input=text,
    text=True,
    stdout=subprocess.PIPE
)

chunks = []
chunk = None

for line in result.stdout.splitlines():
    if line == "EOS":
        for chunk in chunks:
            if "メロス" in chunk["text"]:
                dst = chunk["dst"]
                if dst != -1:
                    print(chunks[dst]["text"])

        chunks = []
        chunk = None

    elif line.startswith("*"):
        parts = line.split()
        dst = int(parts[2].replace("D", ""))

        chunk = {"text": "", "dst": dst}
        chunks.append(chunk)

    else:
        surface, feature = line.split("\t")
        pos = feature.split(",")[0]

        if pos != "記号":
            chunk["text"] += surface
