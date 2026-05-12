from pathlib import Path


def tail(file: Path, n: int) -> None:
    with file.open() as f:
        print("".join([line for line in f][-n::1]))


if __name__ == "__main__":
    text = Path("popular-names.txt")
    tail(text, 10)
