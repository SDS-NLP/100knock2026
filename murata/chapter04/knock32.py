import MeCab

def extract_noun(text):
    mecab = MeCab.Tagger()
    nodes = mecab.parseToNode(text)
    no_list = []
    front = ""
    flag_no = False
    while nodes:
        surface = nodes.surface
        feature = nodes.feature.split(',')
        if (feature[0] == "名詞" and flag_no == False):
            front = surface
        elif (feature[0] == "名詞" and flag_no == True):
            no_list.append([front, surface])
            flag_no = False
        elif (surface == "の" and feature[0] == "助詞"):
            flag_no = True
        else:
            flag_no = False
        nodes = nodes.next
    return no_list

if __name__ == "__main__":
    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """
    results = extract_noun(text)
    for i in range(len(results)):
        result = results[i]
        print(f"{result[0]}の{result[1]}")