#1列目の文字列の出現頻度を求め、出現頻度と名前を出現頻度の多い順に並べて表示せよ。確認にはcut, uniq, sortコマンドを用いよ。

with open("popular-names.txt", "r", encoding = "utf-8") as names:
    
    name_count = {} #カウント用の辞書(key:名前, value:出現回数)を作成
    
    for line in names:
        
        line_list = line.split() #分割してリストへ
        
        if line_list[0] not in name_count: #辞書に名前がないとき
            
            name_count[line_list[0]] = 1 #1つ目としてカウント
             
        else:
            
            name_count[line_list[0]] += 1 #名前がすでにあるときはカウントを加算
            
sorted(name_count.items(), key = lambda name : name[1], reverse = True)