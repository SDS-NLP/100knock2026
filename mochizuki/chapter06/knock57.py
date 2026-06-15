import numpy as np
from sklearn.cluster import KMeans
import knock50

model = knock50.model

COUNTRIES = [
    'Australia', 'Austria', 'Belgium', 'Brazil', 'Canada', 'China', 'Cuba',
    'Denmark', 'Egypt', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'India', 'Ireland', 'Israel', 'Italy', 'Japan', 'Mexico', 'Netherlands',
    'Norway', 'Pakistan', 'Poland', 'Portugal', 'Russia', 'South_Korea',
    'Spain', 'Sweden', 'Switzerland', 'Thailand', 'Turkey', 'Ukraine',
    'United_Kingdom', 'United_States',
]

countries = [c for c in COUNTRIES if c in model]
vectors = np.array([model[c] for c in countries])

if __name__ == '__main__':
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
    labels = kmeans.fit_predict(vectors)

    for country, label in sorted(zip(countries, labels), key=lambda x: x[1]):
        print(f'{label}\t{country}')


"""0       Cuba
0       Egypt
0       Hungary
0       Israel
0       Poland
0       Russia
0       Turkey
0       Ukraine
1       Australia
1       Canada
1       India
1       Mexico
1       United_Kingdom
1       United_States
2       Austria
2       Belgium
2       Denmark
2       Finland
2       Germany
2       Netherlands
2       Norway
2       Sweden
2       Switzerland
3       Brazil
3       France
3       Greece
3       Ireland
3       Italy
3       Portugal
3       Spain
4       China
4       Japan
4       Pakistan
4       South_Korea
4       Thailand"""
