from pathlib import Path

text = Path("popular-names.txt")

with text.open() as f:
    print(
        "\n".join(
            [
                "\t".join(col)
                for col in (
                    sorted(
                        [line.split() for line in f],
                        key=lambda line: int(line[2]),
                        reverse=True,
                    )
                )
            ]
        )
    )
