text = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

word = text.split() #textを各単語に分割

print(word)

count = [] #各単語の文字数を格納するリスト

for i in range(len(word)):
    
    count.append(len(word[i])) #wordリストの先頭からi番目の文字数をcountに追加

print(count)