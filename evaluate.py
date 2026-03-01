import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel("Gen_AI Dataset.xlsx")

queries = df["Query"].unique()

catalog = pd.read_csv("catalog_prepared.csv")

texts = catalog["text"].tolist()
urls = catalog["Assessment_url"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

recalls = []

for query in queries:

    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = scores.argsort()[-10:][::-1]

    predicted = set([urls[i] for i in top_indices])
    actual = set(df[df["Query"] == query]["Assessment_url"])

    recall = len(predicted & actual) / len(actual)
    recalls.append(recall)

mean_recall = sum(recalls) / len(recalls)

print("Mean Recall@10:", round(mean_recall, 3))