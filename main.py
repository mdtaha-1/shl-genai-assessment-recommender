from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- GLOBALS ----------
model = None
df = None
urls = None
embeddings = None


# ---------- FAST START ----------
@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------- LAZY LOADER ----------
def load_resources():
    global model, df, urls, embeddings

    if model is not None:
        return

    print("Loading AI resources...")

    # IMPORT INSIDE FUNCTION (IMPORTANT)
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv("catalog_prepared.csv")
    urls = df["Assessment_url"].tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = np.load("embeddings.npy")

    print("Resources loaded ✅")


# ---------- RECOMMENDER ----------
@app.post("/recommend")
def recommend(query: str):

    load_resources()

    from sklearn.metrics.pairwise import cosine_similarity

    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[-5:][::-1]

    results = []

    for i in top_indices:
        results.append({
            "name": df.iloc[i]["name"],
            "url": urls[i],
            "score": round(float(scores[i]), 3),
            "reason": f"Semantically matched query: {query}"
        })

    return {"recommendations": results}
