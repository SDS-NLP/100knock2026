#1列目の文字列の異なり（文字列の種類）を求めよ。確認にはcut, sort, uniqコマンドを用いよ。

with open("popular-names.txt", "r", encoding = "utf-8") as names:
    
    name_list = []
    
    for line in names:
        
        line_list = line.split()
        
        if line_list[0] not in name_list: #名前がname_listにないとき追加
            
            name_list.append(line_list[0])
            
print(name_list)            