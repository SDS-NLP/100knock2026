def cipher(txt):
    result = ""

    for c in txt:
        if c.islower():
            result += chr(219 - ord(c))
        else:
            result += c

    return result


def main():
    encrypt = cipher("Hello world")
    print(f"暗号化: {encrypt}")
    decrypt = cipher(encrypt)
    print(f"復号: {decrypt}")


main()
