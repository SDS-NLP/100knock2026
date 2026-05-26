import MeCab
import re

import subprocess


def get_pair(text):
    result = subprocess.run(["cabocha", "-f1"], input=text, text=True, capture_output=True)

    chunks = []
    print(result.stdout)

    for line in result.stdout.splitlines():
        if line == "EOS":
            for text, dst in chunks:
                if dst != -1:  # 係り先があるものだけ
                    print(f"{text}\t{chunks[dst][0]}")
            chunks = []  # リセット

        # 係先を取り出す
        elif line.startswith("*"):
            dst = int(line.split()[2][:-1])
            chunks.append(["", dst])

        else:
            if chunks:
                chunks[-1][0] += line.split("\t")[0]


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
