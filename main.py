from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data (lightweight)
df = pd.read_csv("catalog_prepared.csv")
texts = df["text"].tolist()
urls = df["Assessment_url"].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)


@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API Running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.api_route("/recommend", methods=["GET", "POST"])
def recommend(query: str):

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    top_indices = scores.argsort()[-5:][::-1]

    results = []
    for i in top_indices:
        results.append({
            "name": df.iloc[i]["name"],
            "url": urls[i],
            "score": round(float(scores[i]), 3),
            "reason": f"Semantic similarity to query: {query}"
        })

    return {"recommendations": results}

