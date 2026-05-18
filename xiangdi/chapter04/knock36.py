import re
import MeCab
from collections import Counter

with open("/Users/caitlyn/Downloads/kokoro.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r"[、。・「」『』（）()［］【】《》〈〉…—！？!?]", "", text)
text = re.sub(r"《.*?》", "", text)
text = re.sub(r"［＃.*?］", "", text)
text = text.replace("｜", "")
text = text.replace("\r", "")

tagger = MeCab.Tagger()
node = tagger.parseToNode(text)

words = []

while node:
    features = node.feature.split(",")

    if node.surface and features[0] != "補助記号":
        words.append(node.surface)

    node = node.next

counter = Counter(words)

for word, freq in counter.most_common(20):
    print(word, freq)