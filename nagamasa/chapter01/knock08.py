def cipher(text):
    
    char_list = []
    for char in text:
        if ord("a") <= ord(char) <= ord("z"):
            num = 219 - ord(char)
            encrypted_char = chr(num)
            char_list.append(encrypted_char)
        else:
            char_list.append(char)
    
    encrypted_sentence = "".join(char_list)
    
    return encrypted_sentence

if __name__ == "__main__":
    print(cipher("I'm Here"))
    print(cipher(cipher("I'm Here")))