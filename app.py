"""
CineSense - Gradio app (Hugging Face Spaces entry point).

Realistic use case: estimate a film's IMDb rating from its TITLE plus whatever
metadata is available. This is where the two AI blocks integrate at inference:

  NLP block   : if the user does not provide genres, a classical title->genre
                classifier infers them from the title text.
  ML block    : the FusedRatingModel combines structured metadata with the
                NLP-derived title features (latent style + sentiment) and the
                (possibly NLP-inferred) genres to predict the rating.

Run locally:  python app.py     (opens on http://localhost:7860)
"""
import os, sys
import numpy as np
import pandas as pd
import joblib
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from nlp_features import TitleFeaturizer        # noqa: needed for unpickling
from fused_model import FusedRatingModel        # noqa: needed for unpickling

ROOT = os.path.dirname(os.path.abspath(__file__))
GENRES = ["Action", "Animation", "Comedy", "Drama", "Documentary", "Romance", "Short"]
MEDIAN_LOG_VOTES = 3.434   # neutral prior when vote count is unknown (train median)

rating_model = joblib.load(os.path.join(ROOT, "models", "rating_model.joblib"))
genre_model = joblib.load(os.path.join(ROOT, "models", "nlp_genre.joblib"))


def infer_genres(title, threshold=0.40):
    probs = genre_model.predict_proba([title])[0]
    pairs = sorted(zip(GENRES, probs), key=lambda x: -x[1])
    chosen = [g for g, p in pairs if p >= threshold] or [pairs[0][0]]
    return chosen, {g: float(p) for g, p in pairs}


def predict(title, year, length, votes, mpaa, manual_genres):
    title = (title or "").strip()
    if not title:
        return "### Please enter a movie title.", {}, ""

    # NLP block: infer genres from the title when the user gives none.
    used_nlp_genres = False
    if manual_genres:
        active = list(manual_genres)
    else:
        active, _ = infer_genres(title)
        used_nlp_genres = True
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
    md += f"- Genres used: **{', '.join(active)}**"
    md += " *(inferred from title by the NLP block)*\n" if used_nlp_genres else " *(provided by you)*\n"
    if cold_start:
        md += ("- **Cold-start mode**: no vote count given, so a neutral popularity "
               "prior was used and the title's text signal carries more weight.\n")
    note = ("This is an estimate. Ratings of obscure films are inherently noisy "
            "(typical error ~1.3 points); confidence is higher for films with many votes.")
    return md, prob_map, note


EXAMPLES = [
    ["The Last Silent Romance", 2018, 104, None, "None", []],
    ["Robo Death Killer 3000", 2002, 89, 45, "R", []],
    ["A Quiet Documentary About Bees", 2011, 76, None, "None", []],
    ["Love in the Time of Spreadsheets", 1999, 95, 1200, "PG-13", []],
]

with gr.Blocks(title="CineSense") as demo:
    gr.Markdown(
        "# CineSense - movie rating estimator\n"
        "Estimate a film's IMDb rating by **fusing structured metadata (ML) with "
        "title-text NLP**. Leave the genres empty and the NLP block will infer them "
        "from the title and feed them to the rating model."
    )
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(label="Movie title", placeholder="e.g. The Last Silent Romance")
            with gr.Row():
                year = gr.Number(label="Year", value=2010, precision=0)
                length = gr.Number(label="Runtime (min)", value=100, precision=0)
            votes = gr.Number(label="Vote count (leave empty if unknown / unreleased)", value=None, precision=0)
            mpaa = gr.Dropdown(["None", "G", "PG", "PG-13", "R", "NC-17"], value="None", label="MPAA rating")
            manual_genres = gr.CheckboxGroup(GENRES, label="Genres (optional - leave empty to let NLP infer)")
            btn = gr.Button("Estimate rating", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_probs = gr.Label(label="Title -> genre probabilities (NLP block)")
            out_note = gr.Markdown()
    btn.click(predict, [title, year, length, votes, mpaa, manual_genres], [out_md, out_probs, out_note])
    gr.Examples(EXAMPLES, [title, year, length, votes, mpaa, manual_genres])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
