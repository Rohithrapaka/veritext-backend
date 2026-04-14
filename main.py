from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import requests
import json
import time
from functools import lru_cache
from typing import Optional

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

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Rate limit configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 10  # seconds

class TextInput(BaseModel):
    text: str


class AIInput(BaseModel):
    text: str


def call_gemini_api_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    """
    Call Gemini API with exponential backoff retry on rate limits.
    Handles 429 (rate limit), 500 (server error), and connection errors.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    retry_delay = INITIAL_RETRY_DELAY
    
    for attempt in range(max_retries):
        try:
            logger.info(f"API call attempt {attempt + 1}/{max_retries}")
            
            response = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1}
                },
                timeout=20
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limited. Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                    continue
                else:
                    raise HTTPException(
                        status_code=429, 
                        detail=f"API rate limit exceeded after {max_retries} retries. Please try again in a few moments."
                    )
            
            # Handle server errors with retry
            if response.status_code >= 500:
                if attempt < max_retries - 1:
                    logger.warning(f"Server error {response.status_code}. Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                    continue
                else:
                    raise HTTPException(status_code=response.status_code, detail="Gemini API server error")
            
            # Success
            if response.status_code == 200:
                return response.json()
            
            # Other errors
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Gemini API error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout. Waiting {retry_delay}s before retry...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                continue
            raise HTTPException(status_code=504, detail="Gemini API request timed out after multiple retries")
        
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error. Waiting {retry_delay}s before retry...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                continue
            raise HTTPException(status_code=503, detail="Cannot reach Gemini API. Please try again later.")
    
    raise HTTPException(status_code=500, detail="Failed to get response from Gemini API after retries")


# -----------
# Health Check
# -----------

@app.get("/")
def home():
    return {
        "status": "ready",
        "service": "veritext-backend",
        "endpoints": ["/health", "/api/plagiarism-check", "/api/ai-detect"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# -----------
# Plagiarism Check (API-based with retry)
# -----------

@app.post("/api/plagiarism-check")
def plagiarism_check(input: TextInput):
    """
    Analyze text for plagiarism using Gemini API.
    Returns similarity analysis and potential plagiarism indicators.
    Handles rate limiting with automatic retries.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Gemini API key")

    try:
        # Limit text length
        text = input.text[:1000]

        prompt = f"""Analyze this text for plagiarism risk and provide similarity assessment.
        
Return ONLY valid JSON with no markdown, no backticks:
{{"plagiarism_risk": 0, "summary": "...", "reasons": []}}

plagiarism_risk: 0-100 score
summary: brief assessment
reasons: list of concerning patterns if any

Text to analyze:
{text}"""

        data = call_gemini_api_with_retry(prompt)
        
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)

        return {
            "plagiarism_risk": parsed.get("plagiarism_risk", 0),
            "summary": parsed.get("summary", ""),
            "reasons": parsed.get("reasons", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plagiarism check error: {e}")
        raise HTTPException(status_code=500, detail="Plagiarism analysis failed")


# -----------
# AI Detection Endpoint (with retry)
# -----------

@app.post("/api/ai-detect")
def ai_detect(input: AIInput):
    """
    Detect if text was generated by AI using Gemini API.
    Returns probability scores for overall text and individual sentences.
    Handles rate limiting with automatic retries.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Gemini API key")

    try:
        sentences = [
            s.strip() + "."
            for s in input.text.split(".")
            if len(s.strip()) > 5
        ]

        if not sentences:
            return {"overall": 0, "sentences": []}

        prompt = f"""You are an AI content detector. Analyze each sentence and estimate the probability (0-100) that it was written by an AI language model.

Return ONLY valid JSON with no markdown, no backticks, no explanation:
{{"overall":50,"sentences":[{{"text":"...","probability":50}}]}}

Sentences:
{chr(10).join(f"{i+1}. {s}" for i, s in enumerate(sentences))}"""

        data = call_gemini_api_with_retry(prompt)
        
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)

        return {
            "overall": parsed.get("overall", 0),
            "sentences": parsed.get("sentences", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI detection error: {e}")
        raise HTTPException(status_code=500, detail="AI detection failed")
