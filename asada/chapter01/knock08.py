import string


def cipher(text):
    return "".join(
        [chr(219 - ord(i)) if i in string.ascii_letters[0:26] else i for i in text]
    )


text = "Doraemon ha dorayaki ga daisuki."

print(cipher(text))
