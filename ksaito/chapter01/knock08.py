def cipher(s: str):
    encoded = []
    for ch in s:
        if not ch.islower():
            encoded.append(ch)
            continue

        num = 219 - ord(ch)
        encoded.append(chr(num))

    return "".join(encoded)

if __name__ == "__main__":
    message = "I am a NLPer 1111"
    encoded_message = cipher(message)
    decoded_message = cipher(encoded_message)
    print(f"encoded message: {encoded_message}\ndecoded message: {decoded_message}")



