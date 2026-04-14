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
SAPLING_API_KEY = os.getenv("SAPLING_API_KEY")
SAPLING_API_URL = "https://api.sapling.ai/api/v1/aidetect"

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


def heuristic_score(text: str) -> float:
    """
    Advanced heuristic scoring based on AI writing patterns.
    Returns score 0-1 (1 = definitely AI-like patterns).
    """
    score = 0.0
    
    # AI writing pattern checks
    checks = [
        { "pattern": r'\bin conclusion\b', "weight": 0.15 },
        { "pattern": r'\bit is worth noting\b', "weight": 0.12 },
        { "pattern": r'\bfurthermore\b', "weight": 0.08 },
        { "pattern": r'\bin summary\b', "weight": 0.10 },
        { "pattern": r'\bmoreover\b', "weight": 0.08 },
        { "pattern": r'\bultimately\b', "weight": 0.07 },
        { "pattern": r'\bit is important to\b', "weight": 0.10 },
        { "pattern": r'\badditionally\b', "weight": 0.06 },
        { "pattern": r'\bsignificantly\b', "weight": 0.05 },
        { "pattern": r'\bnotably\b', "weight": 0.08 },
        { "pattern": r'\binterestingly\b', "weight": 0.07 },
        { "pattern": r'\bcrucially\b', "weight": 0.06 },
        { "pattern": r'\bessentially\b', "weight": 0.05 },
        { "pattern": r'\bcomprehensively\b', "weight": 0.08 },
        { "pattern": r'\bnoteworthy\b', "weight": 0.07 },
    ]
    
    for check in checks:
        if re.search(check["pattern"], text, re.IGNORECASE):
            score += check["weight"]
    
    # Check sentence length uniformity (AI tends to be uniform)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]
    
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Low variance = likely AI (uniform sentence lengths)
        if variance < 10:
            score += 0.2
    
    # Check for perfect grammar patterns (AI often has very clean text)
    # Count potential grammatical errors (simple heuristic)
    words = text.split()
    if len(words) > 10:
        # Look for common AI-perfect patterns
        if re.search(r'\bthe\s+[aeiou]', text, re.IGNORECASE):  # "the" followed by vowel (proper article usage)
            score += 0.05
        
        # Check for balanced paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 2:
            para_lengths = [len(p.split()) for p in paragraphs if p.strip()]
            if len(para_lengths) >= 2:
                para_avg = sum(para_lengths) / len(para_lengths)
                para_variance = sum((l - para_avg) ** 2 for l in para_lengths) / len(para_lengths)
                if para_variance < 50:  # Similar paragraph lengths
                    score += 0.1
    
    return min(1.0, score)


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
                    "temperature": 0.1,  # Low temperature for consistent, deterministic output
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}  # Force JSON response
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


def _parse_model_score_and_label(parsed: dict, raw_text: str) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[list]]:
    """Extract score, label, confidence, and signals from parsed model JSON or raw text."""
    score = None
    label = None
    confidence = None
    signals = None

    if parsed:
        # Try multiple possible keys for score
        for key in ("score", "ai_probability", "ai_score", "probability"):
            if key in parsed:
                score = parsed.get(key)
                break

        # Try multiple possible keys for label/classification
        for key in ("classification", "label"):
            if key in parsed:
                label = parsed.get(key)
                break

        confidence = parsed.get("confidence")
        signals = parsed.get("signals")

    # Process score
    if isinstance(score, str):
        score = score.strip().replace("%", "")
        try:
            score = float(score)
        except ValueError:
            score = None

    if isinstance(score, (int, float)):
        score = float(score)
        if score > 1.0:
            score = score / 100.0
        score = max(0.0, min(1.0, score))
    else:
        score = None

    # Process label
    if isinstance(label, str):
        label = label.strip()
        # Map to our expected labels
        label_lower = label.lower()
        if "ai generated" in label_lower:
            label = "AI Generated"
        elif "likely ai" in label_lower:
            label = "Likely AI"
        elif "uncertain" in label_lower:
            label = "Uncertain"
        elif "likely human" in label_lower:
            label = "Likely Human"
        elif "human written" in label_lower or "human" in label_lower:
            label = "Human Written"
        else:
            label = None

    # Process confidence
    if isinstance(confidence, str):
        confidence = confidence.strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = None

    # Process signals
    if not isinstance(signals, list):
        signals = None

    return score, label, confidence, signals

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


def _call_sapling_for_ai_detection(text: str) -> Tuple[float, bool, list]:
    """
    Call Sapling AI detection API to get a normalized AI score.
    Returns: (score 0-1, success_bool, signals)
    """
    if not SAPLING_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Sapling API key")

    text_limited = text[:2000]
    payload = {"key": SAPLING_API_KEY, "text": text_limited}
    
    try:
        response = requests.post(SAPLING_API_URL, json=payload, timeout=20)
        if response.status_code != 200:
            logger.warning("Sapling API returned %s: %s", response.status_code, response.text[:1000])
            raise HTTPException(status_code=502, detail="Sapling API error")

        parsed = response.json()
        score = parsed.get("score")

        if isinstance(score, str):
            score = score.strip().replace("%", "")
            try:
                score = float(score)
            except ValueError:
                score = None

        if not isinstance(score, (int, float)):
            logger.error("Sapling response missing numeric score: %s", parsed)
            raise HTTPException(status_code=502, detail="Invalid Sapling API response")

        score = float(score)
        if score > 1.0:
            score = score / 100.0
        score = max(0.0, min(1.0, score))

        return score, True, []

    except requests.exceptions.RequestException as e:
        logger.warning("Sapling API request failed: %s", e)
        raise HTTPException(status_code=503, detail="Cannot reach Sapling AI detection API")


def _combine_signals(text: str, api_score: float, use_api: bool) -> Tuple[float, str, str, dict]:
    """
    Combine multiple signals for final AI detection score.
    Returns: (final_score 0-1, classification, confidence, details)
    """
    # Calculate heuristic scores
    gibberish = is_gibberish(text)
    repetition = repetition_score(text)
    structure = structure_score(text)
    length = text_length_heuristic(text)
    pattern_score = heuristic_score(text)
    
    details = {
        "api_score": api_score,
        "gibberish": round(gibberish, 2),
        "repetition": round(repetition, 2),
        "structure": round(structure, 2),
        "length_signal": round(length, 2),
        "pattern_score": round(pattern_score, 2),
    }
    
    # If text is gibberish, definitely human
    if gibberish > 0.5:
        logger.info(f"Detected gibberish (score: {gibberish})")
        final_score = 0.1
        label = "Human Written"
        confidence = "high"
        return final_score, label, confidence, {**details, "reason": "Gibberish detected"}
    
    # Combine signals with weights
    if use_api:
        # Blend API score (70%) with advanced heuristics (30%)
        heuristic_combined = (
            0.4 * pattern_score +      # AI writing patterns
            0.3 * repetition +          # Repetition patterns
            0.2 * structure +           # Structure uniformity
            0.1 * length                # Length-based signal
        )
        final_score = (0.7 * api_score) + (0.3 * heuristic_combined)
    else:
        # Without API, rely on heuristics only
        final_score = (
            0.45 * pattern_score +
            0.25 * repetition +
            0.20 * structure +
            0.10 * length
        )
    
    # Ensure score is in valid range
    final_score = max(0.0, min(1.0, final_score))
    
    # Determine label and confidence using 5-tier system
    if final_score >= 0.85:
        label = "AI Generated"
        confidence = "high"
    elif final_score >= 0.65:
        label = "Likely AI"
        confidence = "medium"
    elif final_score >= 0.35:
        label = "Uncertain"
        confidence = "low"
    elif final_score >= 0.15:
        label = "Likely Human"
        confidence = "medium"
    else:
        label = "Human Written"
        confidence = "high"
    
    return final_score, label, confidence, details


@app.post("/api/ai-detect")
def ai_detect(input: AIInput):
    """
    Detect if text was generated by AI using Sapling plus heuristics.
    Returns: {classification, confidence, score, signals}
    """
    if not SAPLING_API_KEY:
        raise HTTPException(status_code=500, detail="Missing Sapling API key")

    try:
        text = input.text.strip()
        if not text:
            return {
                "classification": "Uncertain",
                "confidence": "low",
                "score": 0.5,
                "signals": []
            }
        
        # Check cache first
        cache_key = _get_cache_key(text)
        if cache_key in _detection_cache:
            logger.info(f"Cache hit for text (key: {cache_key[:8]}...)")
            return _detection_cache[cache_key]
        
        api_score = 0.0
        use_api = False
        heuristic_signals = []

        try:
            api_score, use_api, _ = _call_sapling_for_ai_detection(text)
        except HTTPException as e:
            logger.warning("Sapling API unavailable, falling back to heuristics: %s", e)
            api_score = 0.0
            use_api = False

        phrase_hits = [
            "in conclusion",
            "it is worth noting",
            "furthermore",
            "in summary",
            "moreover"
        ]
        for phrase in phrase_hits:
            if re.search(re.escape(phrase), text, re.IGNORECASE):
                heuristic_signals.append(phrase)

        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            if variance < 10:
                heuristic_signals.append("uniform sentence length")

        final_score, classification, confidence, details = _combine_signals(text, api_score, use_api)
        
        # Build response
        logger.info("FINAL DETECTION RESULT: score=%s classification=%s confidence=%s use_api=%s", final_score, classification, confidence, use_api)
        result = {
            "classification": classification,
            "confidence": confidence,
            "score": round(final_score, 2),
            "signals": heuristic_signals
        }
        
        # Cache result
        _detection_cache[cache_key] = result
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI detection error: {e}", exc_info=True)
        return {
            "classification": "Uncertain",
            "confidence": "low",
            "score": 0.5,
            "signals": []
        }
