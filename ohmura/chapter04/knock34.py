import spacy
import ginza

nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])

bunsetu_list = list(ginza.bunsetu_spans(nlp("メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。")))

for span in bunsetu_list:
    if "メロス" in span.text:
        head_token = span.root.head
        for b in bunsetu_list:
            if head_token in b:
                if span.text != b.text:
                    print(f"主語：{span.text}\t➔\t述語：{b.text}")
                break