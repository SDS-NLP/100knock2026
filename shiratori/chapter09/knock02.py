from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
text = "The movie was full of [MASK]."

results = unmasker(text, top_k=10)

for i, r in enumerate(results, start=1):
    print(f"{i:2d}: {r['token_str']:15} {r['score']:.6f}")

#  1. fun             0.107119
#  2. surprises       0.066345
#  3. drama           0.044684
#  4. stars           0.027217
#  5. laughs          0.025413
#  6. action          0.019517
#  7. excitement      0.019038
#  8. people          0.018290
#  9. tension         0.015031
# 10. music           0.014646
