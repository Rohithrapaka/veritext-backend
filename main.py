from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppState:
    model = None
    questions = []
    embeddings = None
    ready = False
    error = None

state = AppState()

class TextInput(BaseModel):
    text: str

async def load_resources_async():
    try:
        loop = asyncio.get_event_loop()

        # Import heavy libraries INSIDE the background task, not at module level
        logger.info("Importing libraries...")
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        logger.info("Loading model...")
        state.model = await loop.run_in_executor(
            None, lambda: SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        )

        logger.info("Loading dataset...")
        df = await loop.run_in_executor(None, lambda: pd.read_csv("clean_questions.csv"))
        state.questions = df["question"].tolist()

        logger.info("Loading embeddings...")
        state.embeddings = await loop.run_in_executor(None, lambda: np.load("embeddings.npy"))

        # Store references so endpoint can use them
        state.np = np
        state.cosine_similarity = cosine_similarity

        state.ready = True
        logger.info("✅ Backend ready!")
    except Exception as e:
        state.error = str(e)
        logger.error(f"❌ Load failed: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_resources_async())
    logger.info("🚀 Server started, loading resources in background...")

@app.get("/")
def home():
    return {"status": "ready" if state.ready else "loading", "error": state.error}

@app.get("/health")
def health():
    return {"status": "ok"}  # Always 200 — Render uses this for health checks

@app.post("/api/plagiarism-check")
def plagiarism_check(input: TextInput):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Still warming up, retry in ~60s")

    query_embedding = state.model.encode([input.text])
    similarities = state.cosine_similarity(query_embedding, state.embeddings)[0]
    top_indices = state.np.argsort(similarities)[-5:][::-1]

    return {
        "results": [
            {"matched_text": state.questions[idx], "similarity_score": round(float(similarities[idx]), 4)}
            for idx in top_indices
        ]
    }