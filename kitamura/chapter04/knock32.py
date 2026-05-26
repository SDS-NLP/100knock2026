import MeCab

file = "merosu.txt"
with open(file, "r", encoding="utf-8") as f:
    text = f.read()

def make_words_list(text):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)
    words_list = []
    while node:
        features = node.feature.split(",")
        words_list.append({
            node.surface:features[0]
        })
        node = node.next

    return words_list

def n_no_n(words_list):
    state = 0
    temp_words = ""
    answer_list = []
    for dict in words_list:
        for word, parts in dict.items():
            if state == 0 or state==2:
                if parts == "名詞":
                    state += 1
                    temp_words += word
                    if state == 3:
                        answer_list.append(temp_words)
                        temp_words = ""
                else:
                    state =0


            elif state ==1:
                if word == "の":
                    state = 2
                    temp_words += word
                elif parts == "名詞":
                    state = 1
                    temp_words = ""
                    temp_words += word
                else:
                    state = 0
                    temp_words = ""
            

    return answer_list
    

def main():
    words_list = make_words_list(text)
    answer = n_no_n(words_list)
    print(answer)


if __name__ =="__main__":
    main()     
            

