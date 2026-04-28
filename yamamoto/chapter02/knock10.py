#ファイルの行数をカウントせよ。確認にはwcコマンドを用いよ。

with open("popular-names.txt", "r", encoding = "utf-8") as names: #ファイルの読み込み
    
    count = 0 #行数をカウント
    
    for line in names:
        
        count += 1

print(count)        

#wc popular-names.txt