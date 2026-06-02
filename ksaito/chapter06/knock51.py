from knock50 import load_model

def cosine_similarity(model, vec1, vec2):
    return model.similarity(vec1, vec2)

def main():
    model = load_model()
    similarity = cosine_similarity(model, 'United_States', 'U.S.')
    print("United states と U.S. の類似度")
    print(similarity)


if __name__ == '__main__':
    main()
