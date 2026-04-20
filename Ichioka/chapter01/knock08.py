def cipher(text):
    return ''.join(
        chr(219 - ord(c)) if c.islower() else c
        for c in text
    )

message = "Hello, World! This is a test message."

encrypted = cipher(message)

decrypted = cipher(encrypted)

print("元:", message)
print("暗号化:", encrypted)
print("復号:", decrypted)