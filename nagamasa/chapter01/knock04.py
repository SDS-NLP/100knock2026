str = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can"

one_char_positions = {1, 5, 6, 7, 8, 9, 15, 16, 19}

word_list = str.replace(",","").replace(".","").split(" ")

atom_dic = {}

for i ,word in enumerate(word_list, start=1):
    if i in one_char_positions:
        atom = word[0]
        atom_dic[atom] = i
    else:
        atom = word[:2]
        atom_dic[atom] = i


print(atom_dic)
