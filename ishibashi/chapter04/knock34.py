import spacy

def extract_predicates_for_subject(text, subject_word):
    """指定した主語に対する述語を返す関数"""
    nlp = spacy.load("ja_ginza")
    doc = nlp(text)

    predicates = []
    for token in doc:
        if token.text == subject_word and token.dep_ == "nsubj":
            predicates.append(token.head.text)

    return predicates

if __name__ == "__main__":
    
    text = """
    メロスは激怒した。
    必ず、かの邪智暴虐の王を除かなければならぬと決意した。
    メロスには政治がわからぬ。
    メロスは、村の牧人である。
    笛を吹き、羊と遊んで暮して来た。
    けれども邪悪に対しては、人一倍に敏感であった。
    """

    predicates = extract_predicates_for_subject(text, "メロス")
    print("「メロス」が主語であるとき")
    print(predicates)