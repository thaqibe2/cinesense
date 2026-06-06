"""
llm.py — optional LLM (OpenAI) layer for CineSense.

Adds a transformer/LLM NLP approach on top of the classical models. Two uses:
  1. extract_movie_features(text): turn a free-text plot/idea into STRUCTURED
     inputs (canonical title, genres, year/runtime hints) that feed the existing
     fused rating model. This is real cross-block integration: free text -> LLM
     -> features -> ML prediction.
  2. explain_prediction(...): produce a short, GROUNDED natural-language
     explanation of a prediction, using only facts we pass in (no hallucinated
     box-office numbers, etc.).

Design rules:
  - The API key is read from the OPENAI_API_KEY environment variable. It is
    never hard-coded and never logged. Set it locally / as a Space secret.
  - Every call is wrapped so failures (no key, no network, rate limit) return
    None and the app falls back to the classical pipeline. The LLM is an
    enhancement, not a hard dependency.
  - One short call per user action; nothing scans the dataset.
"""
from __future__ import annotations
import os
import json

GENRES = ["Action", "Animation", "Comedy", "Drama", "Documentary", "Romance", "Short"]
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # or "gpt-4.1-nano" (cheaper)


def llm_available() -> bool:
    """True only if the OpenAI SDK is importable and a key is configured."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
        return True
    except Exception:
        return False


def _client():
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def _chat_json(system: str, user: str, max_tokens: int = 300, temperature: float = 0.2):
    """Call the chat API in JSON mode; return a dict or None on any failure."""
    try:
        resp = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None


def _chat_text(system: str, user: str, max_tokens: int = 160, temperature: float = 0.4):
    try:
        resp = _client().chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def validate_features(raw: dict) -> dict | None:
    """Coerce/validate the LLM JSON into a safe feature dict."""
    if not isinstance(raw, dict):
        return None
    genres = [g for g in (raw.get("genres") or []) if g in GENRES]
    title = str(raw.get("canonical_title") or "").strip()
    out = {"canonical_title": title[:120], "genres": genres}
    try:
        y = int(raw.get("year"))
        out["year"] = y if 1900 <= y <= 2030 else None
    except Exception:
        out["year"] = None
    try:
        r = int(raw.get("runtime_min"))
        out["runtime_min"] = r if 1 <= r <= 600 else None
    except Exception:
        out["runtime_min"] = None
    out["tone"] = str(raw.get("tone") or "")[:80]
    return out


def extract_movie_features(text: str) -> dict | None:
    """Free-text description/idea -> structured features for the ML model."""
    text = (text or "").strip()
    if not text or not llm_available():
        return None
    system = (
        "You convert a movie description into structured metadata. "
        "Respond ONLY with JSON of the form "
        '{"canonical_title": str, "genres": [subset of '
        f"{GENRES}], \"year\": int|null, \"runtime_min\": int|null, \"tone\": str}}. "
        "Pick genres only from the provided list. If a field is unknown use null. "
        "canonical_title is a short plausible film title for the description."
    )
    raw = _chat_json(system, text, max_tokens=200, temperature=0.2)
    return validate_features(raw) if raw else None


def llm_genres_for_title(title: str) -> list | None:
    """LLM-predicted genres for a bare title (used in the comparison study)."""
    title = (title or "").strip()
    if not title or not llm_available():
        return None
    system = (
        "Classify the likely genres of a film GIVEN ONLY ITS TITLE. "
        f"Respond ONLY with JSON {{\"genres\": [subset of {GENRES}]}}. "
        "Choose the most plausible genres; pick at least one."
    )
    raw = _chat_json(system, title, max_tokens=80, temperature=0.0)
    if not raw:
        return None
    return [g for g in (raw.get("genres") or []) if g in GENRES]


def explain_prediction(*, title, predicted_rating, genres_used, genre_probs,
                       cold_start, votes, year, length) -> str | None:
    """Grounded, concise explanation using only the facts provided."""
    if not llm_available():
        return None
    facts = {
        "title": title, "predicted_rating_out_of_10": round(float(predicted_rating), 2),
        "genres_used": genres_used, "genre_probabilities": {k: round(v, 2) for k, v in genre_probs.items()},
        "vote_count_known": (votes is not None and votes > 0), "votes": votes,
        "cold_start": cold_start, "year": year, "runtime_min": length,
    }
    system = (
        "You explain a movie-rating model's prediction to a user in 2-3 sentences. "
        "Use ONLY the facts in the JSON. Do NOT invent box-office, reviews, awards, "
        "cast, or any fact not given. Mention the predicted rating, the main genre "
        "signal, and (if cold_start) that the estimate is uncertain because the vote "
        "count is unknown. Be plain and concise."
    )
    return _chat_text(system, json.dumps(facts), max_tokens=160, temperature=0.4)
