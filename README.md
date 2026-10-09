# Calgary Water Main Replacement Prioritisation — Classical ML

Avathon AI/ML Hiring Challenge · **Track C — Classical ML** · Domain: water supply network reliability
(own domain, framed nearest **S1** — detecting supply disruptions early)

> **In one line:** every 1 January, rank Calgary's 60,740 water pipes by predicted breaks per km so the small
> replacement budget (~0.25% of the network a year) goes where it prevents the most breaks. On five unseen test
> years the model's list catches **17× more breaks than random — about 37 breaks avoided vs 14 for the best
> simple rule**.

## Deliverables

| Deliverable | Link |
|---|---|
| Technical write-up (PDF, 2 pages) | [write-up/writeup.pdf](write-up/writeup.pdf) |
| 5-minute walkthrough video | [Watch on Loom](https://www.loom.com/share/ef116e65eedd4bc69a899acc32ca1abd) |
| Code, results and reproduction steps | this repository |

**Contents:** [Problem](#1-business-problem) · [Data & join](#2-data-and-cleaning) · [EDA](#3-what-the-data-shows) ·
[ML formulation](#4-machine-learning-formulation) · [Evaluation](#5-evaluation-design) · [Experiments](#6-experiments) ·
[Fixes: before vs after](#7-design-fixes-before-vs-after) · [Final results](#8-final-results) ·
[Explainability](#9-explainability-shap) · [Leakage & monitoring](#10-leakage-and-monitoring) ·
[How the team uses it](#11-how-the-team-uses-it) · [Limitations](#12-limitations) · [Reproduce](#13-reproduce) ·
[Structure](#14-repository-structure)

---

## 1. Business problem

- Calgary has **~5,400 km of water mains**. A break means an emergency dig, homes without water and a damaged street.
- Every break is **repaired** (no choice). The choice is which pipes to **replace** before they break.
- Replacement budget: **~8 km in 2025, 15 km/yr planned for 2027–2030** ≈ **0.15–0.3% of the network a year**.
- **Decision:** which pipes go on this year's replacement list? An asset engineer approves the final list.
- **Value, with no invented costs:** **breaks avoided per km replaced**.

## 2. Data and cleaning

City of Calgary Open Data, [Open Government Licence – City of Calgary](https://www.calgary.ca/cs/iis/licensing-city-data/licensing-city-data.html).

| Table | Rows | Key fields |
|---|---|---|
| [Water Main Breaks](https://data.calgary.ca/Environment/Water-Main-Breaks/dpcu-jr23) | 37,538 breaks, 1956 – mid-2026 | date, failure mode, status, location (point) |
| [Public Water Main](https://data.calgary.ca/Environment/Public-Water-Main/w6h9-w33i) | 60,740 segments, 5,419 km | material, diameter, install year, length, pressure zone, route (line) |

### Joining breaks to pipes

The tables share **no ID column**, so they are joined by geometry — each break is attached to the pipe line it sits
on (`src/data.py → link_breaks_to_pipes`).

```
 BREAKS (37,538)                              PIPES (60,740)
 date · failure type · status · point         material · install year · diameter · length · line
          │              no shared ID column              │
          └──────────────────────┬────────────────────────┘
                                 ▼
   1. Convert degrees → metres     x = lng × 111,320 × cos(51°),  y = lat × 110,540
                                 ▼
   2. Spatial index of all pipe lines (shapely STRtree)
                                 ▼
   3. For each break point: nearest pipe line + distance

                    ●  break (0.8 m off the line)
                    ┆
      ══════════════╧══════════════   pipe A: cast iron, 1967   ← nearest → match
      ─────────────────────────       pipe B: PVC, 2003 (14 m away → ignored)
                                 ▼
   4. Check the match
      a) distance: median 0.77 m, 99.9% within 1.27 m; only 2 of 37,538 breaks > 5 m from any pipe
      b) time:     pipe installed ≤ break year  → this is the pipe that broke
                   pipe installed >  break year → the broken pipe was replaced; this is its successor
                                                  (these are the STATUS = RETIRED breaks)
                                 ▼
   5. Use the linked breaks
      ┌──────────────────────────────────┬──────────────────────────────────────────┐
      │ ACTIVE, pipe old enough (15,216) │ RETIRED, pipe since replaced (22,255)    │
      ├──────────────────────────────────┼──────────────────────────────────────────┤
      │ own break history of the pipe    │ not the successor's own history          │
      │ label (breaks in the next year)  │ counted as nearby breaks for surrounding │
      │ nearby breaks for other pipes    │ pipes (known on the planning date)       │
      └──────────────────────────────────┴──────────────────────────────────────────┘
```

**Worked example:** a corrosion break on 1 April 1978 lies 1.0 m from a cast-iron pipe laid in 1967 (258 m) —
distance ✓, time ✓ → it counts in that pipe's history. **Why the time check matters:** without it, a 1975 break
would attach to the PVC pipe laid in 1995 to replace the old one, making new PVC look as if it breaks.

**Other cleaning:** 67 ACTIVE breaks matched to a newer pipe are dropped; segments under 10 m count as 10 m of
exposure; failure modes decoded from metadata (A full circular, B split, C corrosion, D fitting, E joint,
F diagonal crack, G hole, S saddle).

## 3. What the data shows

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (target statistics use pre-2021 data only).

| Finding | Evidence |
|---|---|
| **Material drives risk** | breaks per 100 km per year (2011–15): cast iron **15.9**, ductile iron 9.5–9.9, PVC **0.1** |
| **Repeat offenders** | pipes that broke before (11% of pipes) carried **66%** of the next breaks |
| **Neighbourhood** | never-broken pipes with recent breaks within 150 m broke **4–6× more** |
| **1980s peak** | 13,001 breaks that decade, 9,104 on pipes later replaced — largely young ductile iron (indirect evidence) |
| **Recording changed ~2000** | exact dates and different failure-mode coding after ~2000 |
| **Rare label** | ~99.6% of pipe-years have no break |

## 4. Machine learning formulation

Supervised **count regression (Poisson)**, used for **ranking**.

| Element | Definition |
|---|---|
| Row | one pipe on 1 January of a planning year (annual snapshots 1996–2025, 1.48M pipe-years) |
| Label | breaks on that pipe during the year (all failure modes) |
| Prediction | expected breaks **per km per year**; pipes sorted by it, list cut at the budget |
| Exposure | `log(length × years)` offset (segments < 10 m count as 10 m) |
| Model | **LightGBM**, Poisson objective, 500 trees, learning rate 0.03, 15 leaves, **min leaf 2,000 rows, L2 = 10** |

| Feature group | Features (all from data before 1 January) |
|---|---|
| Pipe | material, pressure zone, diameter, age, log(length) |
| Own history | breaks ever, breaks last 10 yrs, breaks per km (ever, 10 yrs), years since last break |
| Nearby breaks | breaks on other pipes within 50 / 150 / 300 m (incl. since-replaced pipes): count in last 10 yrs + recency-weighted count |

**Why counts, not yes/no:** replacing a pipe that would break 3 times avoids 3 breaks; counts scale with length;
predicted counts add up to a planning number (~180–210 breaks/yr).

## 5. Evaluation design

```
for each year Y:   train on 1996 … Y-1   →   rank pipes on 1 Jan Y   →   score on Y's breaks
validation  Y = 2016–2020   (all choices made here)        test  Y = 2021–2025
```

**Metric — recall at a length budget:** take the top-ranked pipes until they reach **0.25% of network length**
(one year's budget); recall = breaks on those pipes ÷ all breaks, **pooled over 5 years**. **Lift** = recall ÷
budget (random = 1×). Also: recall at 1%, recall on pipes with **no break history**, predicted ÷ actual counts.
Not accuracy (99.6% "no break") and not AUC (averages over budgets no city funds).

## 6. Experiments

All runs in MLflow (export: [`results/mlflow_experiment_log.csv`](results/mlflow_experiment_log.csv)).

**6.1 Model families** (validation, 5 seeds, 5-year design, original settings)

| Model | R@1% | Lift@1% | R@1% no history | Predicted ÷ actual | Verdict |
|---|---|---|---|---|---|
| XGBoost | 8.1% | 8.1× | 5.2% | 1.07 | tied with LightGBM (same family) |
| **LightGBM + nearby** | 7.6% | 7.6× | **7.2%** | **1.03** | **chosen** |
| Poisson GLM | 6.9–7.4% | 6.9–7.4× | 5.9–8.3% | 1.23–1.32 | over-predicts counts |
| Rule: past breaks per km | 6.5% | 6.5× | 0.9% | — | baseline; blind on never-broken pipes |
| Isolation Forest | 6.1% | 6.1× | 0.3% | — | finds *unusual*, not risky, pipes |

Failure-type features added nothing (recording changed ~2000) and were dropped.

**6.2 Evaluation design** (validation): 5-year snapshots and annual-all-history tied within year-to-year noise;
a 4-year sliding window was worse (R@1% 6.9% vs 7.3%) → **annual, all history** chosen (matches the yearly budget).

**6.3 Regularisation tuning** (validation only, [`results/tuning_validation.csv`](results/tuning_validation.csv); 150-tree settings excluded — they over-predict counts 3.7×)

| min leaf | L2 | R@0.25% (breaks / 1,042) | R@1% | In-sample R@1% | Gap | Shortlist < 20 m |
|---|---|---|---|---|---|---|
| 50 *(original)* | 0 | 2.6% (27) | 7.8% | 14.9% | **7.2** | **15.7%** |
| 500 | 0 | 2.0% (21) | 8.5% | 12.6% | 4.1 | 9.4% |
| 2000 | 0 | 2.1% (22) | 7.6% | 11.3% | 3.8 | 7.2% |
| **2000** | **10** | **2.2% (23)** | 7.8% | 12.0% | 4.3 | **7.0%** |

Headlines are within noise (±5 breaks); the original setting memorised training breaks (largest gap, most stubs
at the top), so **min leaf 2000, L2 10** was chosen.

## 7. Design fixes: before vs after

A train-vs-held-out check and a look at the 2026 shortlist exposed two issues: **overfitting** and **nearby-break
features ignoring breaks on since-replaced pipes**. Both were fixed (choices made on validation).

| | Before | **After** |
|---|---|---|
| Validation R@0.25% (lift) | 2.0% (8.2×) | **2.2% (9.0×)** |
| Test R@0.25% (lift) | 3.4% (13.4×) | **4.3% (17.3×)** |
| Test breaks avoided over 5 years | ~29 | **~37** |
| Test R@1% | 9.7% | **10.4%** |
| Test R@1%, never-broken pipes | 11.8% | 11.3% |
| Test predicted ÷ actual | 1.19 | **1.14** |
| In-sample vs validation lift (overfitting) | 27× vs 8.8× | **15.5× vs 8.8×** |
| Top of the 2026 shortlist | 10 m PVC (2002) with no breaks | **cast iron (1957–1967) with past and nearby breaks** |

## 8. Final results

| Split | Recall at 0.25%/yr | Lift | R@1% | R@1% no history | Predicted ÷ actual |
|---|---|---|---|---|---|
| Validation 2016–20 · rule | 1.5% | 6.1× | 6.4% | 0.9% | — |
| Validation 2016–20 · **model** | 2.2% | 9.0× | 7.6% | 9.3% | 1.00 |
| Test 2021–25 · rule | 1.6% | 6.6× | 8.0% | 2.0% | — |
| Test 2021–25 · **model** | **4.3%** | **17.3×** | **10.4%** | **11.3%** | 1.14 |

Train vs held out, same metric ([`results/scores_train_val_test.csv`](results/scores_train_val_test.csv), seed 0):
in-sample lift 15.5× → validation 8.8×; in-sample 24.1× → test 18.2×.

![Breaks caught on the test years by random, rule and model](results/test_recall.png)

**Business impact:** at Calgary's real budget the model's list would have avoided **~37 breaks vs 14** for the rule
over 2021–2025 — **more than double the breaks avoided per km replaced**, roughly four to five fewer emergency
digs a year for the same capital spend. Per-year results are noisy (one year's budget catches 3–14 breaks), so
the honest range across validation and test is **9–17× random**.

**Not a per-pipe prophecy:** even top pipes have a small yearly chance of breaking; the value is that the listed
13.5 km holds many times more breaks per km than the network average.

## 9. Explainability (SHAP)

[`notebooks/02_model_explanations.ipynb`](notebooks/02_model_explanations.ipynb) — SHAP values shown as
**multipliers on a typical pipe's break rate**. Material dominates, then recent breaks on nearby pipes, then the
pipe's own record. Among cast iron, risk falls with age past ~60 years (the oldest survivors are the sturdier
remainder), so "oldest first" is the wrong rule.

![SHAP explanations for three pipes](results/shap_three_pipes.png)

## 10. Leakage and monitoring

**Leakage:** features for year Y use only breaks before 1 January Y; training always ends before the graded year.
[`tests/test_no_leakage.py`](tests/test_no_leakage.py): invent breaks in/after the planning year → features must
not change (verified to fail if one `<` becomes `<=`).

**Feature drift** ([`src/pipeline/drift_report.py`](src/pipeline/drift_report.py), Evidently, validation vs test years): **pipe age**
and **nearby breaks within 300 m** drifted (scores 0.13–0.14 > 0.1). The label was not flagged although the break
rate per pipe-year fell 37% — mostly-zero labels hide rate changes.

**Calibration monitor** ([`src/pipeline/monitor.py`](src/pipeline/monitor.py)): an Evidently test each year — actual breaks within
±15% of predicted; alarm after two consecutive failures.

![Calibration monitor 2016–2025](results/calibration_monitor.png)

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|
| Predicted | 208 | 204 | 208 | 210 | 210 | 206 | 196 | 192 | 189 | 186 |
| Actual | 191 | 233 | 218 | 211 | 189 | 156 | 186 | 195 | 169 | 149 |
| Within ±15% | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |

2021 and 2025 fail but not consecutively, so no alarm yet; a second failing year in 2026 would trigger a retrain
or rescale. The model lags the continuing decline in breaks.

![MLflow run comparison](results/mlflow_run_comparison.jpg)

## 11. How the team uses it

**Once a year, before the capital plan (1 January):**

```
dvc repro                                                    refresh data, rebuild features
python -m src.pipeline.train                                 retrain on every labelled year → new model version in MLflow
python -m src.pipeline.predict --year 2026 --budget 0.0025   → results/shortlist_2026.csv
python -m src.pipeline.monitor                               last year's predicted vs actual; alarm rule
```

**Output — [`results/shortlist_2026.csv`](results/shortlist_2026.csv)** (model `pipe-break-ranker` v2; 168 pipes, 13.5 km):

| Column | Meaning |
|---|---|
| `rank` | priority (1 = replace first) |
| `globalid` | the pipe's ID in the city GIS |
| `material`, `install_year`, `diameter`, `length` | what the pipe is |
| `breaks_all`, `nearby_150m_10y` | its own breaks, and recent breaks around it |
| `rate_per_km_yr` | predicted breaks per km per year — the ranking score |
| `expected_breaks` | predicted breaks on this pipe next year |
| `top_reasons` | the three factors that put it on the list, e.g. `material x4.89; breaks_per_km x1.83; nearby_300m_decay x1.37` |

Only replaceable units are listed (active pipes ≥ 10 m); the list stops at the budget. The network-wide expected
count (179 breaks for 2026) supports crew and repair-budget planning. **The engineer decides**, adding what the
data can't see (planned roadworks, critical customers, access). The model does not pick repairs, price work or
weight breaks by severity.

## 12. Limitations

- **Survivorship:** the inventory is a 2026 snapshot, so historical years only contain pipes that survived to 2026; the worst, already-replaced pipes are missing, which likely flatters the scores.
- **Falling break rates:** the model lags the decline and over-predicts counts (14% on test) — the calibration monitor watches it.
- **Recording changes ~2000:** dates and failure-mode coding changed; failure-type features were dropped.
- **No cost or severity data:** breaks are counted equally; replacement cost assumed proportional to length.
- **Segment-level ranking:** the GIS splits mains into segments; real projects replace runs of connected segments — grouping them is the next step.
- **Location join:** at junctions a break can attach to the neighbouring segment (rare; 99.9% within 1.27 m).

## 13. Reproduce

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Step | Command | Output |
|---|---|---|
| 1 | `dvc repro` | data + feature tables (`dvc.lock` records file hashes) |
| 2 | `notebooks/01_eda.ipynb` | exploration |
| 3 | `python -m src.experiments.compare_models --model lgbm --features nearby` | model-family runs (`rule`, `iforest`, `glm`, `lgbm`, `xgb`) |
| 4 | `python -m src.experiments.tune` | regularisation tuning on validation |
| 5 | `python -m src.experiments.rolling --design annual --window all --model lgbm [--split test]` | final validation / test |
| 6 | `python -m src.experiments.score_table` | train vs validation vs test |
| 7 | `python -m src.pipeline.train` then `python -m src.pipeline.predict --year 2026` | registered model + shortlist |
| 8 | `notebooks/02_model_explanations.ipynb` | SHAP |
| 9 | `python -m src.experiments.plot_results`, `python -m src.pipeline.drift_report`, `python -m src.pipeline.monitor` | charts, drift, calibration monitor |
| 10 | `python -m pytest tests/` | leakage test |
| 11 | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | browse runs |

Python 3.11; dependencies pinned; seeds fixed (stochastic models averaged over seeds).

## 14. Repository structure

Production code (what runs every year) is separate from the experiments that justified each choice.

```
src/
  data.py            load tables, convert to metres, link breaks to nearest pipe
  features.py        pipe × planning-date feature tables (5-year and annual)
  evaluate.py        recall at a length budget, lift, calibration
  model.py           feature sets, frozen settings, LightGBM Poisson model
  pipeline/          ← the yearly production job
    download_data.py   fetch the Calgary datasets
    train.py           train on all labelled years, register model in MLflow
    predict.py         load registered model → shortlist with SHAP reasons
    monitor.py         yearly calibration test (Evidently)
    drift_report.py    feature drift (Evidently)
  experiments/       ← how the choices were made
    compare_models.py  rule, Isolation Forest, GLM, LightGBM, XGBoost + MLflow logging
    tune.py            regularisation tuning on validation
    rolling.py         sliding-window evaluation, pooled metrics
    score_table.py     train vs validation vs test
    plot_results.py    test-recall chart
notebooks/           01 exploration, 02 SHAP explanations
tests/               leakage test
results/             charts, metrics, shortlist, reports, MLflow export
write-up/            technical write-up (PDF + HTML source)
```

Run modules from the repository root, e.g. `python -m src.pipeline.predict`.
