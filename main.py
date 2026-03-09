from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import numpy as np

app = FastAPI()

# Allow frontend access (important for Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model = None
questions = []
embeddings = None


class TextInput(BaseModel):
    text: str


# Load model and dataset on startup
@app.on_event("startup")
def load_resources():
    global model, questions, embeddings

    print("Loading Sentence-BERT model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading dataset...")
    df = pd.read_csv("clean_questions.csv")

    questions = df["question"].tolist()

    print("Generating embeddings...")
    embeddings = model.encode(questions)

    print("Backend ready!")


# Root route (for Render health check)
@app.get("/")
def home():
    return {"message": "VeriText API is running"}


# Main plagiarism check endpoint
@app.post("/api/plagiarism-check")
def plagiarism_check(input: TextInput):

    global model, embeddings, questions

    query_embedding = model.encode([input.text])

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = np.argsort(similarities)[-5:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "matched_text": questions[idx],
            "similarity_score": float(similarities[idx])
        })

    return {
        "results": results
    }