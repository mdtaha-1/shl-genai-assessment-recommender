import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load dataset
df = pd.read_excel("Gen_AI Dataset.xlsx")

# unique assessment URLs
assessments = df["Assessment_url"].unique()

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# create embeddings for assessments
assessment_embeddings = model.encode(assessments)

# user query
query = "Need Java developer with collaboration skills"

# query embedding
query_embedding = model.encode([query])

# similarity scores
scores = cosine_similarity(query_embedding, assessment_embeddings)[0]

# top 5 results
top_indices = scores.argsort()[-5:][::-1]

print("\nTop Recommendations:\n")

for i in top_indices:
    print(assessments[i])