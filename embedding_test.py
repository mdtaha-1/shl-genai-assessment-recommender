from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load model (AI brain)
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "Java developer with teamwork skills",
    "Python data analyst",
    "Sales and communication expert"
]

# convert text to embeddings
embeddings = model.encode(texts)

# compare similarity
score = cosine_similarity([embeddings[0]], [embeddings[1]])

print("Similarity score:", score[0][0])