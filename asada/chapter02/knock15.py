from pathlib import Path


def split(file: Path, n: int) -> None:
    lines = file.read_text().splitlines()
    size = len(lines) // n
    for i in range(n):
        new_f = Path(f"child{i}.txt")
        new_f.write_text(
            "\n".join(lines[size * i : (None if i == n - 1 else size * (i + 1))])
        )


if __name__ == "__main__":
    text = Path("popular-names.txt")
    split(text, 10)
