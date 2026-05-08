# 20. JSONデータの読み込み
# Wikipedia記事のJSONファイルを読み込み、「イギリス」に関する記事本文を表示せよ。
# 問題21-29では、ここで抽出した記事本文に対して実行せよ。

import gzip
import json

with gzip.open("/Users/caitlyn/Downloads/jawiki-country.json.gz", "rt", 
               encoding="utf-8") as f:
    for line in f:
        content = json.loads(line)
        if content["title"] == "イギリス":
            output_text = content["text"]

with open("uk.txt", "w", encoding="utf-8") as output:
    output.writelines(output_text)