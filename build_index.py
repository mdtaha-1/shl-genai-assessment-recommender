import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("catalog_prepared.csv")

texts = df["text"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding embeddings...")
embeddings = model.encode(texts)

np.save("embeddings.npy", embeddings)

print("Saved embeddings.npy")