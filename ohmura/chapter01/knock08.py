def cipher(text):
    result = ""
    for char in text:
        if char.islower():
            result += chr(219 - ord(char))
        else:
            result += char
    return result

message = "I am an NLPer"
encrypted = cipher(message)
print(f"暗号化: {encrypted}")

decrypted = cipher(encrypted)
print(f"復号化: {decrypted}")