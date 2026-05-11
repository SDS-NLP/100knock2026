def cipher(sentence):
    
    result = ""
    
    for i in range(len(sentence)):
        
        print(sentence[i])
        
        if ord(sentence[i]) >= 97: #i番目の文字が小文字のとき
            
            result = result + chr(219 - ord(sentence[i])) #別の文字に変換して結合
        else:
            
            result = result + sentence[i] #大文字のときはそのまま結合
        
        print(result)           
            
    return result

sentence = "The pen is mightier than the sword"
print(cipher(sentence))                    