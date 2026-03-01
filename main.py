from fastapi import FastAPI
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ----------------------------
# CORS (frontend access)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# GLOBAL VARIABLES (loaded at startup)
# ----------------------------
df = None
urls = None
model = None
embeddings = None


# ----------------------------
# LOAD EVERYTHING AFTER SERVER STARTS
# (Fixes Render timeout issue)
# ----------------------------
@app.on_event("startup")
def load_resources():
    global df, urls, model, embeddings

    print("Loading data...")
    df = pd.read_csv("catalog_prepared.csv")
    urls = df["Assessment_url"].tolist()

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading embeddings...")
    embeddings = np.load("embeddings.npy")

    print("Startup complete ✅")


# ----------------------------
# ROOT ROUTE
# ----------------------------
@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API Running"}


# ----------------------------
# HEALTH CHECK (Render friendly)
# ----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}


# ----------------------------
# RECOMMENDATION API
# ----------------------------
@app.post("/recommend")
def recommend(query: str):

    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = scores.argsort()[-5:][::-1]

    results = []

    for i in top_indices:
        results.append({
            "name": df.iloc[i]["name"],
            "url": urls[i],
            "score": round(float(scores[i]), 3),
            "reason": f"Recommended because it semantically matches the query: {query}"
        })

    return {"recommendations": results}
