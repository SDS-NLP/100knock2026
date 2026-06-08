import knock50

if __name__ == '__main__':
    results = knock50.model.most_similar(
        positive=['Spain', 'Athens'], negative=['Madrid'], topn=10
    )
    for word, score in results:
        print(f'{word}\t{score:.4f}')


"""Ioannis_Drymonakos      0.5553
Greeks  0.5451
Ioannis_Christou        0.5401
Hrysopiyi_Devetzi       0.5248
Heraklio        0.5208
Athens_Greece   0.5169
Lithuania       0.5167
Iraklion        0.5147"""