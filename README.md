---
title: CineSense
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
pinned: false
license: mit
---

# CineSense - Movie Rating Estimator (ML Numeric + NLP)

CineSense estimates a film's IMDb rating by **fusing structured metadata
(machine learning on numeric data) with title-text NLP**. It is built for the
AI Applications module and combines two blocks in a single, integrated system
rather than running them side by side.

## What makes it different
Most rating predictors stop at "predict a number". CineSense instead studies a
concrete integration question: *can the words in a film's title add predictive
signal on top of its metadata, and when does that matter?* A classical NLP
model turns the raw title into genre probabilities + latent style/sentiment
features, and those NLP outputs are fed as inputs into the numeric rating model.

## Blocks
- **ML Numeric (E.1)** - HistGradientBoosting / RandomForest / Ridge predict
  rating from year, runtime, log-votes, genre flags and MPAA.
- **NLP (E.2)** - classical NLP on titles: char n-gram TF-IDF + logistic
  regression for multi-label genre prediction, plus a TitleFeaturizer
  (char-TFIDF -> SVD + VADER sentiment + word-rarity + style stats).
- **LLM (optional, OpenAI)** - a prompt-engineered layer: describe a film in
  free text and the LLM extracts structured features for the ML model, and it
  writes a grounded explanation of each prediction. Enabled by setting
  `OPENAI_API_KEY`; without it the app cleanly falls back to the classical models.
- **Integration** - the NLP title features and (optionally NLP-inferred) genres
  are concatenated with the numeric features for a single fused regressor. At
  inference, if the user gives no genres, the NLP block infers them from the
  title and hands them to the ML block.

## Data sources (all offline / reproducible)
1. IMDb `movies` table (pydataset) - numeric metadata.
2. IMDb `movies` table - the `title` free text (NLP role).
3. VADER sentiment lexicon (vaderSentiment) - title tone features.
4. English word-frequency corpus (wordfreq) - title word-rarity features.

## Headline results (held-out test set)
| Model | feature set | RMSE | MAE | R2 |
|---|---|---|---|---|
| mean baseline | - | 1.543 | - | 0.00 |
| HistGBM | numeric only | 1.318 | 0.988 | 0.270 |
| HistGBM | title NLP only | 1.517 | 1.195 | 0.034 |
| HistGBM | **fused** | **1.290** | **0.969** | **0.301** |

When votes and genres are unknown (a new/unreleased film), fusion improves RMSE
by **+1.95%** over numeric-only, because the title becomes the richest available
signal. Title->genre NLP reaches test micro-F1 0.40.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python run_all.py        # data prep -> EDA -> NLP -> numeric -> evaluation
python app.py            # serves the Gradio UI on http://localhost:7860
```

## Project layout
```
src/data_prep.py      data loading, cleaning, feature engineering, splits
src/eda.py            exploratory analysis figures
src/nlp_features.py   TitleFeaturizer (NLP -> numeric features)
src/train_nlp.py      title -> genre classifier comparison
src/fused_model.py    deployable FusedRatingModel wrapper
src/train_numeric.py  model comparison + numeric/text/fused ablation
src/evaluate.py       sparse-metadata scenario, importance, error & fairness
app.py                Gradio app (Hugging Face Spaces entry point)
models/               saved .joblib artifacts (created by training)
reports/              metrics JSON + figures
docs/documentation.md filled module documentation template
```

See `docs/documentation.md` for the full module write-up.
