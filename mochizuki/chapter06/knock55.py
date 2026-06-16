import knock54

SEMANTIC = {'capital-world', 'capital-common-countries', 'currency', 'city-in-state', 'family'}

sections = knock54.get_sections()

sem_correct = sem_total = syn_correct = syn_total = 0

for section, words in sections.items():
    results = knock54.run_analogy(words)
    correct = sum(1 for r in results if r[3].lower() == r[4].lower())
    if section in SEMANTIC:
        sem_correct += correct
        sem_total += len(results)
    else:
        syn_correct += correct
        syn_total += len(results)

if __name__ == '__main__':
    print(f'Semantic:  {sem_correct}/{sem_total} = {sem_correct/sem_total:.4f}')
    print(f'Syntactic: {syn_correct}/{syn_total} = {syn_correct/syn_total:.4f}')
