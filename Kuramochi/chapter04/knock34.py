from knock33 import extract_dependency_relations

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""


def find_predicates_of_melos(text):
    predicates = []
    for src, dst in extract_dependency_relations(text):
        if 'メロスは' in src or 'メロスが' in src:
            predicates.append(dst)
    return predicates


if __name__ == '__main__':
    for predicate in find_predicates_of_melos(text):
        print(predicate)