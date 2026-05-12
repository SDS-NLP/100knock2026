def chipher(S):
    S = list(S)
    for i in range(len(S)):
        s = S[i]
        if (97 <= ord(s) < 123):
            S[i] = chr(219-ord(s))
    return "".join(S)