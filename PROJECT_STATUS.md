# CineSense - Project Status & Requirements Audit

_Last updated: 2026-06-02. Submission deadline: 2026-06-07, 18:00._

Legend: **DONE** = complete in this repo · **OPEN** = needs your action (cannot be done for you) · **IMPROVED** = was added/strengthened in the latest pass.

## TL;DR - what's left for you (3 actions)
1. **OPEN - Create the GitHub repo and push this folder**, then add collaborators **`jasminh`** and **`bkuehnis`** (Settings -> Collaborators).
2. **OPEN - Deploy to Hugging Face Spaces** (SDK: Gradio, entry `app.py`; the README already has the Space config header). Copy the live URL.
3. **OPEN - Paste both URLs** into `docs/documentation.md` (Project Metadata + Deployment) and tick the two checkboxes there; replace the provisional app image (`reports/figures/13_app_preview.png`) with a real screenshot of the live Space.

Everything else below is done.

---

## A. General Project Requirements
| Requirement | Status | Evidence |
| --- | --- | --- |
| Combine >=2 blocks | **DONE** | ML Numeric + NLP |
| Blocks meaningfully integrated (conceptual + technical) | **DONE** | NLP title features + inferred genres feed the rating model; see `src/fused_model.py`, pipeline figure `reports/figures/00_pipeline.png` |
| Multiple and different data sources | **DONE / IMPROVED** | 4 sources: IMDb numeric, IMDb title text, VADER lexicon, **wordfreq corpus (added this pass)** |
| Sources not used in the semester (not Zurich apartments, not dog breeds) | **DONE** | IMDb movies, VADER, wordfreq - all new |
| Well-motivated, realistic use case | **DONE** | Cold-start rating estimation from a title + sparse metadata |
| Completed independently and documented | **DONE** | `docs/documentation.md` |

## B. Documentation Requirements (template fully completed)
| Item | Status | Notes |
| --- | --- | --- |
| 1. Project idea & methodology | **DONE** | Sections 1.1-1.2, incl. pipeline diagram |
| 2. Data & preprocessing (incl. EDA with key findings) | **DONE / IMPROVED** | Sources tables; **explicit EDA findings + figures embedded this pass** (2A.2) |
| 3. Modeling & implementation | **DONE** | Model selection, comparison + iteration tables, libraries/structure |
| 4. Evaluation & analysis | **DONE** | Metrics, ablation, error analysis, cold-start + sparse-metadata |
| 5. Deployment (working URL + screenshots) | **PARTIAL / OPEN** | App is built and runs; **URL + real screenshot need you to deploy** |
| 6. Execution instructions | **DONE** | Section 4, `run_all.py` |
| Template structure unchanged | **DONE** | Used the official template headings verbatim |

## C. Assessment Criteria
| Criterion | Status | Notes |
| --- | --- | --- |
| Clarity | **DONE** | Concise docs, diagram, embedded figures |
| Technical correctness | **DONE** | Leakage columns (`r1..r10`) removed; seeded splits; pinned versions |
| Depth of analysis | **DONE / IMPROVED** | Ablation, permutation importance, vote-bucket + sparse-metadata, fairness |
| Quality of integration | **DONE / IMPROVED** | Fusion gain over numeric-only roughly doubled to ~2.2% after adding rarity features |
| Clean, reproducible implementation | **DONE** | Offline data, `run_all.py`, ~1 min on 2 CPUs, no GPU |
| Bonus (3rd block / extended eval / ethics) | **PARTIAL** | No 3rd block; **extended eval + ethics/fairness done** (bonus section 5) |

## D. Submission
| Item | Status | Notes |
| --- | --- | --- |
| Submit as GitHub link by 2026-06-07 18:00 | **OPEN** | You must create + push the repo |
| Add `jasminh` and `bkuehnis` to the repo | **OPEN** | Add as collaborators after creating the repo |

## E.1 ML Numeric Data
| Requirement | Status | Evidence |
| --- | --- | --- |
| >=1 structured/numeric dataset | **DONE** | IMDb movies (58,733 rows) |
| EDA (distributions, relationships, anomalies) | **DONE / IMPROVED** | `src/eda.py`, findings now in docs 2A.2 |
| Feature engineering / selection / transformation | **DONE** | log transforms, indicators, decade, n_genres, scaling, OHE |
| Train & compare >=2 models | **DONE** | Ridge vs RandomForest vs HistGBM |
| Quantitative evaluation with appropriate metrics | **DONE** | RMSE/MAE/R2 on held-out test |
| Interpretation + error analysis | **DONE** | Largest residuals, vote-bucket RMSE |
| Explain how the numeric model uses/produces cross-block I/O | **DONE** | Consumes NLP title features + inferred genres |

## E.2 NLP
| Requirement | Status | Evidence |
| --- | --- | --- |
| Clear definition of text data | **DONE** | Movie titles (free text) |
| NLP-specific preprocessing and/or prompt design | **DONE** | char/word TF-IDF, SVD, sentiment, rarity |
| >=1 NLP approach | **DONE** | Classical NLP (TF-IDF + OVR LogReg / NB) |
| >=1 comparison of models/prompts/retrieval | **DONE** | 3 approaches compared |
| Qualitative and/or quantitative evaluation | **DONE / IMPROVED** | F1 metrics + **qualitative outputs & failure cases** (`reports/nlp_qualitative.md`) |
| Explain how NLP integrates with the other block | **DONE** | Provides features + inferred genres to the ML model |

## E.3 Computer Vision
**N/A** - not selected (only 2 blocks required). A 3rd block was intentionally skipped to keep the two selected blocks tightly integrated rather than adding an unrelated model side by side.

---

## Improvements made in this pass
1. **Added a 4th, genuinely distinct data source** (`wordfreq` English word-frequency corpus) and derived two title word-rarity features. This strengthened the "multiple and different data sources" requirement **and** improved results: fused test RMSE 1.303 -> **1.290** (R2 0.286 -> 0.301); fusion gain over numeric-only roughly doubled (~1.1% -> ~2.2%); sparse-metadata gain 1.33% -> **1.95%**.
2. **Added a clear pipeline/architecture diagram** (`reports/figures/00_pipeline.png`) for documentation section 1.2.
3. **Added NLP qualitative analysis** (`reports/nlp_qualitative.md`): representative predictions + real failure cases, as the template's 2B.5 requests.
4. **Embedded all key figures inline** in `docs/documentation.md` and **added explicit EDA findings** (distributions, weak correlations, genre/decade effects, missing-data and low-vote anomalies).
5. **Added `LICENSE` (MIT)** to match the README declaration and updated `run_all.py` to include the qualitative step.
6. **Added an optional OpenAI LLM layer** (`src/llm.py`): free-text movie descriptions -> structured features for the ML model, plus grounded AI explanations of predictions, with a classical-vs-LLM comparison script (`src/llm_compare.py`). This adds a second NLP approach (LLM/prompt engineering) and an explicit "AI" component; it falls back cleanly when no `OPENAI_API_KEY` is set.
6. Refreshed every metric in `docs/documentation.md` and `README.md` to the retrained values.

## Honest caveats / known limitations
- **Deployment screenshot is provisional**: the live Space could not be created from here, so `13_app_preview.png` is a faithful rendering of the app's output, not a screenshot of a running deployment. Replace it after deploying.
- **Fusion gain is real but modest** (~2.2% RMSE). This is expected - a 16-character title carries limited signal - and is reported honestly; the value concentrates where metadata is missing (cold-start / sparse-metadata).
- **No non-Latin-script titles** exist in the dataset, so the model is untested on them (documented under fairness).
- Two of the four data-source entries are the IMDb table used in two roles (numeric vs text). The template explicitly allows this ("if the same source is used twice for different roles, add it twice"), and the VADER and wordfreq sources are fully independent.
