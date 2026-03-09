from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
import uvicorn
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

# --- State container ---
class AppState:
    model = None
    questions = []
    embeddings = None
    ready = False
    loading = False
    error = None

state = AppState()


class TextInput(BaseModel):
    text: str


async def load_resources_async():
    """Load heavy ML resources in the background after server starts."""
    state.loading = True
    try:
        logger.info("Loading Sentence-BERT model...")
        # Run blocking I/O in thread pool so event loop stays free
        loop = asyncio.get_event_loop()
        state.model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        )
        logger.info("Model loaded.")

        logger.info("Loading dataset...")
        df = await loop.run_in_executor(
            None,
            lambda: pd.read_csv("dataset.csv")
        )
        state.questions = df["question"].tolist()
        logger.info(f"Dataset loaded: {len(state.questions)} questions.")

        logger.info("Loading embeddings...")
        state.embeddings = await loop.run_in_executor(
            None,
            lambda: np.load("embeddings.npy")
        )
        logger.info("Embeddings loaded.")

        state.ready = True
        logger.info("✅ Backend fully ready!")

    except Exception as e:
        state.error = str(e)
        logger.error(f"❌ Failed to load resources: {e}")
    finally:
        state.loading = False


@app.on_event("startup")
async def startup_event():
    """
    Kick off background loading WITHOUT blocking port binding.
    The server becomes available immediately; resources load in background.
    """
    asyncio.create_task(load_resources_async())
    logger.info("🚀 Server started. Resources loading in background...")


# --- Health & readiness endpoints (Render uses GET / for health checks) ---

@app.get("/")
def home():
    return {
        "message": "VeriText API is running",
        "status": "ready" if state.ready else ("loading" if state.loading else "error"),
        "error": state.error
    }


@app.get("/health")
def health():
    """Render health check endpoint."""
    if state.ready:
        return {"status": "healthy"}
    if state.error:
        # Return 200 so Render doesn't kill the instance during error recovery
        return JSONResponse(status_code=200, content={"status": "error", "detail": state.error})
    return JSONResponse(status_code=200, content={"status": "loading"})


@app.get("/ready")
def readiness():
    """Returns 503 until model is fully loaded — useful for frontend checks."""
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="Service is warming up. Try again in a moment."
        )
    return {"status": "ready"}


@app.post("/api/plagiarism-check")
def plagiarism_check(input: TextInput):
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please retry in 30–60 seconds."
        )

    query_embedding = state.model.encode([input.text])
    similarities = cosine_similarity(query_embedding, state.embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]

    results = [
        {
            "matched_text": state.questions[idx],
            "similarity_score": round(float(similarities[idx]), 4)
        }
        for idx in top_indices
    ]
    return {"results": results}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
```

---

### Render Configuration Changes

**Start Command** (in Render dashboard → Settings):
```
python main.py
```
This is fine. Alternatively, the more production-correct form is:
```
uvicorn main:app --host 0.0.0.0 --port $PORT