import CaboCha

text = """
メロスは激怒した。
"""

pumpkin = CaboCha.Parser()
result = pumpkin.parse(text).toString(CaboCha.FORMAT_TREE)
print(result)
