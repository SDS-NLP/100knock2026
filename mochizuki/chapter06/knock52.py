import knock50

if __name__ == '__main__':
    for word, score in knock50.model.most_similar('United_States', topn=10):
        print(f'{word}\t{score:.4f}')

"""United_Sates    0.7401
U.S.    0.7311
theUnited_States        0.6404
America 0.6178
UnitedStates    0.6167
Europe  0.6133
countries       0.6045
Canada  0.6019"""