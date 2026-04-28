def cipher(text):
    return "".join([chr(219 - ord(i)) if i.islower() else i for i in text])


text = "Doraemon ha dorayaki ga daisuki."

print(cipher(text))
