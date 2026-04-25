file = "popular-names.txt"
names_pop = {}

with open(file, "r", encoding = "utf-8") as f:
    for line in f:
        cols = line.split()
        name = cols[0]
        if name in names_pop:
            names_pop[name] += 1

        else:
            names_pop[name] = 1

sorted_name = sorted(names_pop.items(), key=lambda x: x[1])

print(sorted_name)
        