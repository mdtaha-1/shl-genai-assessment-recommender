# SHL GenAI Assessment Recommendation System

## Overview
This project implements a semantic similarity–based assessment recommendation engine that suggests relevant SHL assessments from natural language recruiter queries. Instead of keyword matching, the system retrieves results based on semantic meaning using vectorized text representations and cosine similarity ranking. The goal is to simulate a real recruiter workflow where hiring requirements are automatically mapped to the most relevant SHL assessments.

---

## Live Demo (Deployed API)
**Base URL:**  
https://shl-genai-assessment-recommender-pd9i.onrender.com

**Interactive API Docs (Swagger UI):**  
https://shl-genai-assessment-recommender-pd9i.onrender.com/docs

**Frontend Deloyment:**
https://shl-frontend-lac.vercel.app/

Example query:
Need Java developer with collaboration skills

The API returns top assessment recommendations ranked by semantic relevance.

---

## Problem Statement
Recruiters typically describe hiring needs using natural language. This project solves the problem of recommending relevant SHL assessments by computing semantic similarity between recruiter queries and assessment descriptions.

---

## Approach
The recommendation engine follows a retrieval-based NLP pipeline:

Recruiter Query → Text Vectorization → Cosine Similarity Search → Top-N Recommendations

The system converts both recruiter queries and assessment descriptions into vectors and ranks results based on similarity scores.

---

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy

---

## Key Features
- Semantic similarity–based retrieval
- Vectorized text representation
- Cosine similarity ranking
- FastAPI backend with Swagger UI
- Cloud deployed API (Render)
- Lightweight, deployment-friendly architecture

---

## API Endpoints

### Root Endpoint
GET /

Response:
{"message":"SHL Assessment Recommendation API Running"}

---

### Health Check
GET /health

Response:
{"status":"healthy"}

---

### Recommendation Endpoint
POST /recommend?query=<recruiter_query>

Example:
Need Java developer with collaboration skills

Returns:
- Top 5 recommended assessments
- Similarity scores
- Matching explanation

---

## Example API Output
{
  "recommendations": [
    {
      "name": "Java 8 New",
      "score": 0.186
    }
  ]
}

---

## Evaluation
**Mean Recall@10:** 0.849

This indicates strong retrieval effectiveness for ranking relevant assessments within the provided dataset.

---

## Local Setup

### 1️⃣ Clone Repository
git clone https://github.com/mdtaha-1/shl-genai-assessment-recommender  
cd shl-genai-assessment-recommender

### 2️⃣ Install Dependencies
pip install -r requirements.txt

### 3️⃣ Run API
uvicorn main:app --reload

### 4️⃣ Open API Docs
http://127.0.0.1:8000/docs

---

## Project Structure
main.py — FastAPI backend  
catalog_prepared.csv — Prepared assessment catalog  
embeddings.npy — Precomputed vectors  
requirements.txt  
README.md

---

## Design Notes
- Designed for scalability to larger SHL catalogs.
- Focused on stable retrieval performance and deployment efficiency.
- Architecture supports future upgrades to transformer-based embeddings if required.

---

## Author
Mohammed Taha Salemin  
GitHub: https://github.com/mdtaha-1
