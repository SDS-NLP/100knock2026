#ファイルの先頭10行に対して、タブ1文字につきスペース1文字に置換して出力せよ。確認にはsedコマンド、trコマンド、もしくはexpandコマンドなどを用いよ。

with open("popular-names.txt", "r", encoding = "utf-8") as names:
    
    count = 0
    
    for line in names:
        
        print(line.replace("\t", " "))
        count += 1
        
        if count >= 10:
            
            break        

#head popular-names.txt | tr "\t" " "        