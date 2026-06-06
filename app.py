"""
CineSense - Gradio app (Hugging Face Spaces entry point).

Estimate a film's IMDb rating from its TITLE plus whatever metadata is available.
Three AI components integrate at inference:

  ML block        : FusedRatingModel combines structured metadata with NLP title
                    features to predict the rating.
  Classical NLP   : a title->genre classifier fills genres when none are given.
  LLM (optional)  : if an OpenAI key is configured, the user can describe a movie
                    in free text -> the LLM extracts structured features
                    (title/genres/year) that feed the ML model, and it writes a
                    grounded natural-language explanation of the prediction.
                    Without a key, the app falls back to the classical pipeline.

Run locally:  python app.py     (opens on http://localhost:7860)
Enable the LLM: set the OPENAI_API_KEY environment variable (Space secret).
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from nlp_features import TitleFeaturizer        # noqa: needed for unpickling
from fused_model import FusedRatingModel        # noqa: needed for unpickling
import llm                                       # optional OpenAI layer (safe without a key)

ROOT = os.path.dirname(os.path.abspath(__file__))
GENRES = ["Action", "Animation", "Comedy", "Drama", "Documentary", "Romance", "Short"]
MEDIAN_LOG_VOTES = 3.434   # neutral prior when vote count is unknown (train median)
LLM_ON = llm.llm_available()

rating_model = joblib.load(os.path.join(ROOT, "models", "rating_model.joblib"))
genre_model = joblib.load(os.path.join(ROOT, "models", "nlp_genre.joblib"))


def infer_genres(title, threshold=0.40):
    probs = genre_model.predict_proba([title])[0]
    pairs = sorted(zip(GENRES, probs), key=lambda x: -x[1])
    chosen = [g for g, p in pairs if p >= threshold] or [pairs[0][0]]
    return chosen, {g: float(p) for g, p in pairs}


def predict(title, description, year, length, votes, mpaa, manual_genres):
    title = (title or "").strip()
    description = (description or "").strip()

    # LLM block: free-text description -> structured features that feed the model.
    llm_features = llm.extract_movie_features(description) if description else None
    if llm_features:
        if not title and llm_features["canonical_title"]:
            title = llm_features["canonical_title"]
        if llm_features["year"]:
            year = llm_features["year"]
        if llm_features["runtime_min"]:
            length = llm_features["runtime_min"]

    if not title:
        msg = "### Please enter a movie title"
        msg += " (or a description, with an OpenAI key set)." if not LLM_ON else " or a description."
        return msg, {}, ""

    # Genre source priority: your checkboxes > LLM-extracted > classical title model.
    if manual_genres:
        active, genre_source = list(manual_genres), "provided by you"
    elif llm_features and llm_features["genres"]:
        active, genre_source = llm_features["genres"], "extracted from your description by the LLM"
    else:
        active, _ = infer_genres(title)
        genre_source = "inferred from the title by the classical NLP block"
    _, prob_map = infer_genres(title)

    # Cold-start handling: unknown votes -> neutral prior.
    cold_start = votes is None or float(votes) <= 0
    log_votes = MEDIAN_LOG_VOTES if cold_start else float(np.log1p(votes))

    row = {
        "title": title, "year": int(year), "length": int(length),
        "log_votes": log_votes, "log_budget": np.nan, "has_budget": 0,
        "n_genres": len(active), "decade": int(int(year) // 10 * 10), "mpaa": mpaa or "None",
    }
    for g in GENRES:
        row[g] = 1 if g in active else 0
    pred = float(rating_model.predict(pd.DataFrame([row]))[0])

    stars = "*" * int(round(pred)) + "." * (10 - int(round(pred)))
    md = f"## Predicted IMDb rating: **{pred:.2f} / 10**\n\n`{stars}`\n\n"
    md += f"- Title used: **{title}**\n"
    md += f"- Genres used: **{', '.join(active)}** *({genre_source})*\n"
    if llm_features:
        md += "- The LLM read your description and produced the structured inputs above.\n"
    if cold_start:
        md += ("- **Cold-start mode**: no vote count given, so a neutral popularity "
               "prior was used and the title's text signal carries more weight.\n")

    # LLM block: grounded explanation (falls back to a fixed note without a key).
    explanation = llm.explain_prediction(
        title=title, predicted_rating=pred, genres_used=active, genre_probs=prob_map,
        cold_start=cold_start, votes=(None if cold_start else int(votes)),
        year=int(year), length=int(length),
    )
    if explanation:
        note = "**AI explanation (LLM):** " + explanation
    else:
        note = ("This is an estimate. Ratings of obscure films are inherently noisy "
                "(typical error ~1.3 points); confidence is higher for films with many votes.")
        if not LLM_ON:
            note += ("\n\n_Tip: set an `OPENAI_API_KEY` to enable AI-written explanations "
                     "and free-text movie descriptions._")
    return md, prob_map, note


EXAMPLES = [
    ["The Last Silent Romance", "", 2018, 104, None, "None", []],
    ["Robo Death Killer 3000", "", 2002, 89, 45, "R", []],
    ["", "A lonely lighthouse keeper befriends a stranded whale over one winter; quiet, melancholic, gorgeous.", 2021, 110, None, "None", []],
    ["Love in the Time of Spreadsheets", "", 1999, 95, 1200, "PG-13", []],
]

_llm_banner = ("**AI explanation: ON** (OpenAI configured)" if LLM_ON
               else "_AI explanation is OFF - set an `OPENAI_API_KEY` secret to enable the LLM features._")

with gr.Blocks(title="CineSense") as demo:
    gr.Markdown(
        "# CineSense - movie rating estimator\n"
        "Estimate a film's IMDb rating by **fusing structured metadata (ML) with "
        "title-text NLP**. Leave genres empty and the NLP block infers them. "
        "With an OpenAI key, you can also **describe a movie in plain text** and get "
        "an **AI-written explanation**.\n\n" + _llm_banner
    )
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="Movie title", placeholder="e.g. The Last Silent Romance")
            description = gr.Textbox(
                label="...or describe the movie (optional, needs OpenAI key)",
                placeholder="e.g. A heist comedy set in a failing circus", lines=2)
            with gr.Row():
                year = gr.Number(label="Year", value=2010, precision=0)
                length = gr.Number(label="Runtime (min)", value=100, precision=0)
            votes = gr.Number(label="Vote count (leave empty if unknown / unreleased)", value=None, precision=0)
            mpaa = gr.Dropdown(["None", "G", "PG", "PG-13", "R", "NC-17"], value="None", label="MPAA rating")
            manual_genres = gr.CheckboxGroup(GENRES, label="Genres (optional - leave empty to let NLP infer)")
            btn = gr.Button("Estimate rating", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_probs = gr.Label(label="Title -> genre probabilities (classical NLP block)")
            out_note = gr.Markdown()
    inputs = [title, description, year, length, votes, mpaa, manual_genres]
    btn.click(predict, inputs, [out_md, out_probs, out_note])
    gr.Examples(EXAMPLES, inputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
