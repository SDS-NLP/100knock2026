from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

model_path = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"
analogy_path = "questions-words.txt"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

country_sections = {
    "capital-common-countries",
    "capital-world",
}

countries = set()
section = None

with open(analogy_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line == "":
            continue

        if line.startswith(":"):
            section = line[2:]
            continue

        if section not in country_sections:
            continue

        word1, word2, word3, word4 = line.split()

        countries.add(word2)
        countries.add(word4)

country_names = []
country_vectors = []

for country in sorted(countries):
    if country in model:
        country_names.append(country)
        country_vectors.append(model[country])

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=111,
    n_init="auto"
)

labels = kmeans.fit_predict(country_vectors)

clusters = {}

for country, label in zip(country_names, labels):
    if label not in clusters:
        clusters[label] = []

    clusters[label].append(country)


for label in sorted(clusters):
    countries_in_cluster = sorted(clusters[label])
    print(f"cluster {label}: {', '.join(countries_in_cluster)}")

# cluster 0: Afghanistan, Algeria, Bahrain, Bangladesh, Bhutan, Cuba, Ecuador, Egypt, Honduras, Iran, Iraq, Jordan, Kyrgyzstan, Laos, Lebanon, Libya, Mauritania, Morocco, Nepal, Nicaragua, Oman, Pakistan, Peru, Qatar, Syria, Tajikistan, Tunisia, Turkmenistan, Uzbekistan, Venezuela
# cluster 1: Australia, Belgium, Canada, Chile, China, Denmark, England, Finland, France, Georgia, Germany, Greenland, Indonesia, Ireland, Italy, Japan, Norway, Philippines, Portugal, Samoa, Spain, Sweden, Switzerland, Taiwan, Thailand, Tuvalu, Uruguay, Vietnam
# cluster 2: Burundi, Eritrea, Liberia, Niger, Rwanda, Somalia, Sudan
# cluster 3: Albania, Armenia, Austria, Azerbaijan, Belarus, Bulgaria, Croatia, Cyprus, Estonia, Greece, Hungary, Kazakhstan, Latvia, Liechtenstein, Lithuania, Macedonia, Malta, Moldova, Montenegro, Poland, Romania, Russia, Serbia, Slovakia, Slovenia, Turkey, Ukraine
# cluster 4: Angola, Bahamas, Belize, Botswana, Dominica, Fiji, Gabon, Gambia, Ghana, Guinea, Guyana, Jamaica, Kenya, Madagascar, Malawi, Mali, Mozambique, Namibia, Nigeria, Senegal, Suriname, Uganda, Zambia, Zimbabwe