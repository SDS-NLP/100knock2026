def cipher(text):
    result = ""
    for c in text:
        if c.islower():
            result += chr(219 - ord(c))
        else:
            result += c
    return result
print(cipher("Hello, World!"))