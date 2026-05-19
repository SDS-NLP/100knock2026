import spacy
import ginza

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def extract_melos_predicates(text):
    config = {
        "components": {
            "compound_splitter": {
                "split_mode": "A",
            }
        }
    }

    nlp = spacy.load("ja_ginza", config=config)
    doc = nlp(text)

    results = []

    for token in doc:
        if token.text == "メロス" and token.dep_ == "nsubj":
            head_token = token.head

            predicate_span = ginza.bunsetu_span(head_token)

            results.append(predicate_span.text)

    return results


if __name__ == "__main__":
    print(extract_melos_predicates(text))
