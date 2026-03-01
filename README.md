# SHL GenAI Assessment Recommendation System

## Overview

This project implements an embedding-based recommendation engine that suggests relevant SHL assessments from natural language recruiter queries.

The system uses semantic similarity between query embeddings and assessment embeddings to retrieve the most relevant results.

---

## Tech Stack

- Python
- FastAPI
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Scikit-learn
- NumPy / Pandas

---

## System Pipeline

Query → Sentence Embedding → Cosine Similarity Search → Top Assessment Recommendations

---

## Key Features

- Semantic (meaning-based) retrieval instead of keyword matching
- Precomputed embeddings for faster runtime performance
- FastAPI backend with interactive docs
- Simple web frontend for testing
- Evaluation using Mean Recall@10

---

## Evaluation

**Mean Recall@10:** `0.849`

---

## Run the API

```bash
uvicorn main:app --reload
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

---

## Data Note

The prototype uses the provided dataset and extracted SHL assessment URLs as the catalog seed.
The architecture is designed to scale to the full catalog when complete access is available.
