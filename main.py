from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Initialize FastAPI
app = FastAPI()

# CORS (IMPORTANT for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading dataset...")
df = pd.read_csv("dataset.csv")
dataset_texts = df["text"].tolist()

print("Loading embeddings...")
dataset_embeddings = np.load("embeddings.npy")

print("API Ready 🚀")


# Request model
class InputText(BaseModel):
    text: str


# Root endpoint (optional but helpful)
@app.get("/")
def home():
    return {"message": "VeriText API is running"}


# Plagiarism check endpoint
@app.post("/api/plagiarism-check")
def check_plagiarism(data: InputText):

    input_text = data.text

    # Convert input text to embedding
    input_embedding = model.encode([input_text])

    # Compute similarity
    scores = cosine_similarity(input_embedding, dataset_embeddings)[0]

    # Get top 5 matches
    top_indices = scores.argsort()[-5:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "matched_text": dataset_texts[idx],
            "similarity_score": float(scores[idx])
        })

    return {
        "matches": results
    }