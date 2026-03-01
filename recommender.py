import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading catalog...")

df = pd.read_csv("catalog_prepared.csv")

texts = df["text"].tolist()
urls = df["Assessment_url"].tolist()

print("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
embeddings = model.encode(texts)

# ----- USER QUERY -----
query = "Need Java developer with collaboration skills"

query_embedding = model.encode([query])

scores = cosine_similarity(query_embedding, embeddings)[0]

top_indices = scores.argsort()[-5:][::-1]

print("\nTop Recommendations:\n")

for i in top_indices:
    print(urls[i])
    