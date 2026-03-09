from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        loop = asyncio.get_event_loop()

        logger.info("Loading model...")
        state.model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        )

        logger.info("Loading dataset...")
        df = await loop.run_in_executor(
            None,
            lambda: pd.read_csv("dataset.csv")
        )

        state.questions = df["text"].tolist()

        logger.info("Loading embeddings...")
        state.embeddings = await loop.run_in_executor(
            None,
            lambda: np.load("embeddings.npy")
        )

        state.np = np
        state.cosine_similarity = cosine_similarity
        state.ready = True

        logger.info("Backend ready!")

    except Exception as e:
        state.error = str(e)
        logger.error(f"Load failed: {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_resources_async())
    logger.info("Server started, loading resources in background...")


@app.get("/")
def home():
    return {
        "status": "ready" if state.ready else "loading",
        "error": state.error
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/plagiarism-check")
def plagiarism_check(input: TextInput):

    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="Backend still warming up. Try again shortly."
        )

    query_embedding = state.model.encode([input.text])

    similarities = state.cosine_similarity(query_embedding, state.embeddings)[0]

    top_indices = state.np.argsort(similarities)[-5:][::-1]

    results = [
        {
            "matched_text": state.questions[idx],
            "similarity_score": round(float(similarities[idx]), 4)
        }
        for idx in top_indices
    ]

    return {"results": results}