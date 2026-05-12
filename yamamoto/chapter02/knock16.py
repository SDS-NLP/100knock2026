#ファイルを行単位でランダムに並び替えよ（注意: 各行の内容は変更せずに並び替えよ）。同様の処理をshufコマンドで実現せよ。

import random

with open("popular-names.txt", "r", encoding = "utf-8") as names:
    
    lines = names.readlines()
    
    shuffle_lines = random.sample(lines, 10)
    
    for line in shuffle_lines:
        
        print(line)
        
#shuf "popular-names.txt"