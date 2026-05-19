import MeCab
import re

import subprocess


def get_pair(text):
    result = subprocess.run(["cabocha", "-f1"], input=text, text=True, capture_output=True)

    lines = result.stdout.splitlines()
    chunks = []

    for line in lines:
        if line == "EOS":
            if chunks:
                for text, dst in chunks:
                    if dst != -1:
                        if "メロス" in text:
                            print(f"{text}\t{chunks[dst][0]}")
            chunks = []

        elif line.startswith("*"):
            parts = line.split()
            dst = int(parts[2][:-1])
            chunks.append(["", dst])

        else:
            surface = line.split("\t")[0]
            if chunks:
                chunks[-1][0] += surface


if __name__ == "__main__":
    text = """
        メロスは激怒した。
        必ず、かの邪智暴虐の王を除かなければならぬと決意した。
        メロスには政治がわからぬ。
        メロスは、村の牧人である。
        笛を吹き、羊と遊んで暮して来た。
        けれども邪悪に対しては、人一倍に敏感であった。
        """

    get_pair(text)
