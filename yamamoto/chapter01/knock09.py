import random

def Typoglycemia(sentence):
    
    word = sentence.split() #文字列を単語に分割
    result = ""
    
    for i in range(len(word)):
        
        if len(word[i]) > 4: #単語の長さが4より大きいとき
            
            a_head = word[i][0] #先頭の文字
            a = word[i][1:len(word[i])-1] #中間の文字列
            a_tail = word[i][len(word[i])-1] #末尾の文字
            a_shuffled = "".join(random.sample(a, len(a))) #中間の文字列をシャッフルし、空の文字列に結合
            result = result + a_head + a_shuffled + a_tail + " " #それぞれを結合し、末尾にスペースを追加
        else:
            
            result = result + word[i] + " " #そのまま結合し、末尾にスペースを追加
    
    return result

sentence = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind."

print(Typoglycemia(sentence))