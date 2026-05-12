import re

def split_words(text):
    """文を単語に分ける関数"""
    words = re.findall(r"\w+", text)
    
    return words

if __name__ == "__main__":
    text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
    words = split_words(text)
    letters_cnt = []
    
    for word in words:
        cnt = len(word)
        letters_cnt.append(cnt)
    
    print(letters_cnt)