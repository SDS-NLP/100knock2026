with open("popular-names.txt", encoding="utf-8") as f:
    head_10 = f.readlines()[:10]

for line in head_10:
    print(line.replace("\t", " "))

# https://qiita.com/nagisa_O/items/1180abb9acc06aa115f0
# head -10 popular-names.txt | tr '\t' ' '

# https://qiita.com/shunbaba/items/faf0cf2dbfd4e205e945
# head -10 popular-names.txt | sed 's/\t/ /g'

# https://eng-entrance.com/linux-command-expand
# https://wa3.i-3-i.info/word11675.html
# head -10 popular-names.txt | expand -t 1
