# Parkinson Motor Score Prediction — ENS Data Challenge #159

Predicting the true OFF motor score for Parkinson's disease patients, from longitudinal clinical data (repeated visits per patient across two cohorts).

## Key insight: longitudinal structure

The dataset has 55,603 rows but only 6,971 unique patients (4 to 12 visits each, chronologically ordered). No patient overlaps between train and test. This has two consequences:

- **Validation must use `GroupKFold` on `patient_id`** — a naive random split leaks visits from the same patient across train/validation and biases the local score.
- **Feature engineering can exploit a patient's other visits** — per-patient statistics, progression trend, and neighboring-visit values, all computed independently for train and test.

## Approach

**Feature engineering**
- Per-patient stats on `off`/`on` (mean/std/min/max) — `off` correlates at 0.885 with the target but is missing ~42% of the time; imputing with the patient's own other visits instead of a global median preserves most of that signal
- Progression slope per patient (`off_slope`/`on_slope`, linear fit vs. age)
- Deviation of the current visit from the patient's baseline
- Previous/next visit value (`off_prev`/`off_next`, `on_prev`/`on_next`)
- Visit number, age at first visit, time since first visit
- Standard ratios/interactions (dose per year, on/off ratio, age × disease duration)

**Models**: LightGBM, XGBoost, CatBoost (native NaN handling — no global imputation on the boosting models), blended (60% / 10% / 30%, weights found by grid search on out-of-fold predictions).

## Results

| Version | Validation | RMSE |
|---|---|---|
| Linear Regression | GroupKFold (patient_id) | 4.85 |
| Random Forest | GroupKFold (patient_id) | 4.19 |
| XGBoost (original feature set) | naive split (biased) | 7.37 |
| + patient-level stats & trajectory features | GroupKFold | 4.35 |
| + progression slope, deviation from baseline | GroupKFold | 4.04 |
| + previous/next visit value | GroupKFold | 3.93 |
| **Ensemble (LightGBM + XGBoost + CatBoost)** | GroupKFold | **~3.91** |

Leaderboard: 145/175 → 78/175 after the first round of fixes (patient-level features + honest validation), further improved locally since.

Hyperparameter tuning (Optuna) was tested and brought negligible improvement (~4.06 vs ~4.04) — the feature engineering is what drives the gain here, not the hyperparameters.

## Setup

```
pip install -r requirements.txt
```

Open `notebooks/p_analysis.ipynb` in Jupyter.

## Data

ENS Data Challenge #159 — clinical data from Parkinson's disease patients (challengedata.ens.fr). Features include medication dosage (LEDD), ON/OFF motor scores, cohort and gene mutation status.

## License

MIT
MIT
