from pathlib import Path


def head(file: Path, n: int) -> None:
    with file.open() as f:
        print("".join(line for _, line in zip(range(n), f)))


if __name__ == "__main__":
    text = Path("popular-names.txt")
    head(text, 10)
