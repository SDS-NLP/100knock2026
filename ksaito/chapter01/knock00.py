def interleave_strings(str1, str2) -> str:
    interleaved = []
    for ch1, ch2 in zip(str1, str2, strict=True):
        interleaved.extend([ch1, ch2])

    return "".join(interleaved)


if __name__ == "__main__":
    a = "パトカー"
    b = "タクシー"
    print(interleave_strings(a, b))

