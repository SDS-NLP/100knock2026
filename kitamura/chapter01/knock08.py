def cipher(text):
    answer = ""
    for character in text:
        code = ord(character)
        if 97 <= code <= 122:
            answer += chr(219-code)

        else:
            answer += character

    return answer
    

text = "I am a SDS student. Are you Hitotsbashi-university student?"
print(cipher(text))