from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global App State
# -----------------------------

class AppState:
    model = None
    questions = []
    embeddings = None
    np = None
    cosine_similarity = None
    ready = False
    error = None

state = AppState()

class TextInput(BaseModel):
    text: str


# -----------------------------
# Load resources in background
# -----------------------------

async def load_resources_async():
    try:
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        loop = asyncio.get_event_loop()

        logger.info("Loading SentenceTransformer model...")

        state.model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        )

        logger.info("Model loaded")

        logger.info("Loading dataset...")

        df = await loop.run_in_executor(
            None,
            lambda: pd.read_csv("dataset.csv")
        )

        state.questions = df["text"].tolist()

        logger.info(f"Dataset loaded: {len(state.questions)} records")

        logger.info("Loading embeddings...")

        state.embeddings = await loop.run_in_executor(
            None,
            lambda: np.load("embeddings.npy")
        )

        logger.info("Embeddings loaded")

        state.np = np
        state.cosine_similarity = cosine_similarity

        state.ready = True

        logger.info("Backend ready!")

    except Exception as e:
        state.error = str(e)
        logger.error(f"Startup failed: {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_resources_async())
    logger.info("Server started — loading AI resources in background...")


# -----------------------------
# Health Endpoints
# -----------------------------

@app.get("/")
def home():
    return {
        "status": "ready" if state.ready else "loading",
        "error": state.error
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Plagiarism Endpoint
# -----------------------------

@app.post("/api/plagiarism-check")
def plagiarism_check(input: TextInput):

    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="Model still loading. Please try again shortly."
        )

    try:

        # ⚠️ Limit extremely long documents
        text = input.text[:5000]

        query_embedding = state.model.encode([text])

        similarities = state.cosine_similarity(
            query_embedding,
            state.embeddings
        )[0]

        # Faster top-5 search
        top_indices = state.np.argpartition(similarities, -5)[-5:]
        top_indices = top_indices[state.np.argsort(similarities[top_indices])][::-1]

        results = []

        for idx in top_indices:
            results.append({
                "matched_text": state.questions[idx],
                "similarity_score": round(float(similarities[idx]), 4)
            })

        return {"results": results}

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Plagiarism analysis failed"
        )