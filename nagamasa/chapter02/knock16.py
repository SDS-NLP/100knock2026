import random

# 16. ランダムに各行を並び替える

with open("popular-names.txt") as f:
    lines = f.readlines()

random.shuffle(lines)

with open("shuffle.txt", "w") as shuffled_file:
    shuffled_file.writelines(lines)

# 確認コマンド(mac): gshuf popular-names.txt > shuffled.txt
# 確認コマンド(GNU Linux): shuf popular-names.txt > shuffled.txt
# shufはGNUコマンド。macOSにはデフォルトで入っていない
# brew install coreutilsでgshufとして使える
# sort -Rでも代替できる

# random.shuffle(list)は破壊的メソッド
# 元のリストを直接書き換えてNoneを返す
# writelines(random.shuffle(lines))はNoneを渡すのでエラーになる
# 必ず二行に分けて書くこと

# random.sample(list, len(list))は非破壊的
# 元のリストを変更せず新しいシャッフル済みリストを返す
# 元のリストを後で使う必要がない場合はshuffleで十分

# >はリダイレクト。stdoutをファイルに書き出す
# > は上書き
# >> は追記
# < はファイルを標準入力として渡す