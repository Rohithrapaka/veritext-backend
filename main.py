from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import requests
import json
import time
import re
import hashlib
from functools import lru_cache
from typing import Optional, Dict, Tuple

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
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Rate limit configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 10  # seconds

# Detection cache to avoid repeated API calls
_detection_cache: Dict[str, dict] = {}

# =========================================
# HEURISTIC FUNCTIONS FOR AI DETECTION
# =========================================

def is_gibberish(text: str) -> float:
    """
    Detect if text is gibberish/nonsensical.
    Returns score 0-1 (1 = definitely gibberish).
    """
    if not text or len(text.split()) < 2:
        return 0
    
    words = text.split()
    # Count words that don't match basic letter patterns
    nonsense_count = 0
    for w in words:
        # Remove punctuation
        clean = re.sub(r'[^a-zA-Z0-9]', '', w)
        # Check if it looks like a word (contains vowels or common patterns)
        if clean and not re.search(r'[aeiouAEIOU]', clean) and len(clean) > 3:
            nonsense_count += 1
    
    gibberish_ratio = nonsense_count / len(words) if words else 0
    return min(1.0, gibberish_ratio)


def repetition_score(text: str) -> float:
    """
    Measure repetition patterns. AI text tends to repeat structure.
    Returns score 0-1 (1 = high repetition).
    """
    if not text:
        return 0
    
    words = text.lower().split()
    if len(words) < 5:
        return 0
    
    # Count unique words vs total
    unique_ratio = len(set(words)) / len(words)
    # High repetition (low unique ratio) = potentially AI
    repetition = 1.0 - unique_ratio
    return min(1.0, repetition * 1.5)  # Scale up slightly


def structure_score(text: str) -> float:
    """
    Measure text structure. Perfect structure = potentially AI.
    Returns score 0-1 (1 = very structured/AI-like).
    """
    lines = text.split('\n')
    sentences = re.split(r'[.!?]+', text)
    
    # Check for very consistent sentence lengths (AI trait)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0
    
    avg_len = sum(lengths) / len(lengths)
    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
    
    # Low variance = potentially AI (score close to 1)
    # High variance = likely human (score close to 0)
    structure = 1.0 / (1.0 + variance)  # Normalize with sigmoid-like function
    
    return structure


def text_length_heuristic(text: str) -> float:
    """
    Very short text is likely human.
    Returns signal 0-1 (1 = potentially AI).
    """
    word_count = len(text.split())
    if word_count < 10:
        return 0.1  # Short text = likely human
    elif word_count < 50:
        return 0.2
    else:
        return 0.3  # Longer text slightly more likely to be AI


class AIInput(BaseModel):
    text: str


class TextInput(BaseModel):
    text: str


def call_grok_api_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    """
    Call Grok API with exponential backoff retry on rate limits.
    Handles 429 (rate limit), 500 (server error), and connection errors.
    """
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    retry_delay = INITIAL_RETRY_DELAY
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Grok API call attempt {attempt + 1}/{max_retries}")
            
            response = requests.post(
                url,
                headers=headers,
                json={
                    "model": "grok-beta",
                    "messages": [
                        {"role": "system", "content": "You are a neutral AI text classifier. Return ONLY valid JSON with no explanation or markdown. Be conservative - do NOT assume text is AI-generated. Random, messy, or incoherent text is likely human. Only score high if you are highly confident."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
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
                    raise HTTPException(status_code=response.status_code, detail="Grok API server error")
            
            # Success
            if response.status_code == 200:
                return response.json()
            
            # Other errors
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Grok API error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout. Waiting {retry_delay}s before retry...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                continue
            raise HTTPException(status_code=504, detail="Grok API request timed out after multiple retries")
        
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error. Waiting {retry_delay}s before retry...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                continue
            raise HTTPException(status_code=503, detail="Cannot reach Grok API. Please try again later.")
    
    raise HTTPException(status_code=500, detail="Failed to get response from Grok API after retries")


def _extract_json_from_response(raw_text: str) -> dict:
    """Safely extract JSON object from raw LLM text."""
    if not raw_text:
        return {}

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try to find the first JSON object in the text
    first_brace = raw_text.find('{')
    last_brace = raw_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw_text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fallback: try regex-style extraction of any JSON-like content
    match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', raw_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


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
    Analyze text for plagiarism risk using Grok API.
    Returns similarity analysis and potential plagiarism indicators.
    Handles rate limiting with automatic retries.
    """
    if not GROK_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Grok API key")

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

        data = call_grok_api_with_retry(prompt)
        
        raw = data["choices"][0]["message"]["content"]
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

def _get_cache_key(text: str) -> str:
    """Generate cache key from text hash."""
    return hashlib.md5(text.encode()).hexdigest()


def _call_grok_for_ai_detection(text: str) -> Tuple[float, bool, Optional[str]]:
    """
    Call Grok API to get AI detection score.
    Returns: (score 0-1, success_bool, model_label)
    """
    # Limit text length for API stability
    text_limited = text[:2000]
    
    prompt = f"""You are a neutral AI text classifier. Analyze this text and return ONLY a JSON object with no markdown, no backticks, no explanation.

Return format: {{"ai_probability": <number between 0 and 1>, "label": "AI|Human|Uncertain", "reasoning": "<brief reason>"}}

Rules:
- Do NOT assume text is AI-generated
- Random, messy, or incoherent text is likely HUMAN (score: 0-0.3)
- Normal conversational text with typos/casual language = HUMAN (score: 0-0.4)  
- Simple, short text = HUMAN (score: 0-0.3)
- Only score HIGH (>0.7) if you see CLEAR AI patterns:
  * Perfectly polished academic writing with zero errors
  * Repetitive structure and formal tone in casual contexts
  * Generic corporate/marketing language with buzzwords
  * Text that sounds like a typical AI assistant output

Text to classify:
{text_limited}"""
    
    try:
        data = call_grok_api_with_retry(prompt, max_retries=2)

        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("RAW MODEL RESPONSE: %s", raw_content[:1000])

        clean = raw_content.replace("```json", "").replace("```", "").strip()
        parsed = _extract_json_from_response(clean)

        if not parsed:
            logger.error("Parsed response is empty or malformed. Raw: %s", raw_content[:1000])
            return 0.5, False, None

        logger.info("PARSED MODEL RESPONSE: %s", parsed)

        raw_score = parsed.get("ai_probability", 0.5)
        raw_label = parsed.get("label", "Uncertain")

        if isinstance(raw_score, str):
            raw_score = raw_score.strip().replace("%", "")
            try:
                raw_score = float(raw_score)
            except ValueError:
                logger.error("Invalid ai_probability format: %s", raw_score)
                raw_score = 0.5

        score = float(raw_score)
        if score > 1.0:
            score = score / 100.0

        score = max(0.0, min(1.0, score))
        if isinstance(raw_label, str):
            raw_label = raw_label.strip()

        logger.info("Normalized AI score: %s, model label: %s", score, raw_label)

        return score, True, raw_label

    except HTTPException as e:
        logger.warning("API error (will use heuristics only): %s", e)
        return 0.5, False, None
    except Exception as e:
        logger.error("Unexpected error calling Grok: %s", e, exc_info=True)
        return 0.5, False, None


def _combine_signals(text: str, llm_score: float, use_llm: bool) -> Tuple[float, str, dict]:
    """
    Combine multiple signals for final AI detection score.
    Returns: (final_score 0-1, label "AI"/"Uncertain"/"Human", details)
    """
    # Calculate heuristic scores
    gibberish = is_gibberish(text)
    repetition = repetition_score(text)
    structure = structure_score(text)
    length = text_length_heuristic(text)
    
    details = {
        "llm_score": llm_score,
        "gibberish": round(gibberish, 2),
        "repetition": round(repetition, 2),
        "structure": round(structure, 2),
        "length_signal": round(length, 2),
    }
    
    # If text is gibberish, definitely human
    if gibberish > 0.5:
        logger.info(f"Detected gibberish (score: {gibberish})")
        return 0.1, "Human", {**details, "reason": "Gibberish detected"}
    
    # Combine signals with weights
    if use_llm:
        # LLM is the main signal, heuristics provide context
        final_score = (
            0.70 * llm_score +           # LLM is primary
            0.10 * repetition +          # AI tends to repeat
            0.10 * structure +           # AI has structured patterns
            0.10 * length                # Very short = usually human
        )
    else:
        # Without LLM, rely on heuristics
        final_score = (
            0.30 * repetition +
            0.30 * structure +
            0.25 * length +
            0.15 * gibberish
        )
    
    # Determine label based on final score
    if final_score > 0.85:
        label = "AI"
    elif final_score > 0.60:
        label = "Uncertain"
    else:
        label = "Human"
    
    return final_score, label, details


@app.post("/api/ai-detect")
def ai_detect(input: AIInput):
    """
    Detect if text was generated by AI.
    Uses hybrid detection: Grok LLM + heuristics.
    Returns: {label: "AI"/"Uncertain"/"Human", score: 0-1, details: {...}, sentences: [...]}
    """
    if not GROK_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Grok API key")

    try:
        text = input.text.strip()
        if not text:
            return {
                "label": "Uncertain",
                "score": 0.5,
                "confidence": "low",
                "message": "Empty text provided",
                "details": {},
                "sentences": []
            }
        
        # Check cache first
        cache_key = _get_cache_key(text)
        if cache_key in _detection_cache:
            logger.info(f"Cache hit for text (key: {cache_key[:8]}...)")
            return _detection_cache[cache_key]
        
        # Get LLM score and optional model label
        llm_score, use_llm, model_label = _call_grok_for_ai_detection(text)
        
        # Combine with heuristics
        final_score, label, details = _combine_signals(text, llm_score, use_llm)
        if model_label:
            details["model_label"] = model_label
        
        # Process sentences for granular results
        sentences = [
            s.strip() + "."
            for s in text.split(".")
            if len(s.strip()) > 5
        ]
        
        sentence_results = []
        for sent in sentences:
            # Use heuristics for individual sentences (no extra API calls)
            sent_gibberish = is_gibberish(sent)
            sent_repetition = repetition_score(sent)
            sent_structure = structure_score(sent)
            
            # Simple heuristic score for sentences
            if sent_gibberish > 0.5:
                sent_score = 0.1
            else:
                sent_score = (
                    0.4 * sent_structure +
                    0.3 * sent_repetition +
                    0.3 * sent_gibberish
                )
            
            # Determine label for sentence
            if sent_score > 0.75:
                sent_label = "AI"
                suspicious = True
            elif sent_score > 0.55:
                sent_label = "Uncertain"
                suspicious = True
            else:
                sent_label = "Human"
                suspicious = False
            
            sentence_results.append({
                "text": sent,
                "probability": round(sent_score, 2),
                "label": sent_label,
                "suspicious": suspicious
            })
        
        # Build response
        logger.info("FINAL DETECTION RESULT: score=%s label=%s use_llm=%s", final_score, label, use_llm)
        result = {
            "label": label,
            "score": round(final_score, 2),
            "confidence": "high" if (final_score > 0.85 or final_score < 0.15) else "medium" if (final_score > 0.60 or final_score < 0.40) else "low",
            "details": details,
            "sentences": sentence_results
        }
        
        # Cache result
        _detection_cache[cache_key] = result
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI detection error: {e}", exc_info=True)
        return {
            "label": "Uncertain",
            "score": 0.5,
            "confidence": "low",
            "message": f"Detection failed: {str(e)[:100]}",
            "details": {},
            "sentences": []
        }
