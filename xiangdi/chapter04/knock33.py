import subprocess

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

def run_cabocha(input_text, *args):
    result = subprocess.run(
        ["cabocha", *args],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def parse_lattice(lattice):
    sentences = []
    chunks = []
    chunk = None

    for line in lattice.splitlines():
        if line == "EOS":
            if chunks:
                sentences.append(chunks)
                chunks = []
            chunk = None
        elif line.startswith("* "):
            parts = line.split()
            link = parts[2]
            link = int(link[:-1]) if link.endswith("D") else int(link)

            chunk = {"link": link, "morphs": []}
            chunks.append(chunk)
        else:
            surface, feature = line.split("\t", 1)
            chunk["morphs"].append((surface, feature.split(",")))

    return sentences


def chunk_text(chunk):
    return "".join(
        surface
        for surface, features in chunk["morphs"]
        if features[0] != "記号"
    )

lattice = run_cabocha(text, "-f1")
sentences = parse_lattice(lattice)

for chunks in sentences:
    for chunk in chunks:
        if chunk["link"] != -1:
            src = chunk_text(chunk)
            dst = chunk_text(chunks[chunk["link"]])
            print(f"{src}\t{dst}")