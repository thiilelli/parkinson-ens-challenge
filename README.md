# Parkinson Motor Score Prediction — ENS Data Challenge #159

Predicting the true OFF motor score for Parkinson's disease patients, from longitudinal clinical data (repeated visits per patient across two cohorts).

## Key insight: longitudinal structure

The dataset has 55,603 rows but only 6,971 unique patients (4 to 12 visits each, chronologically ordered). No patient overlaps between train and test. This has two consequences:

- **Validation must use `GroupKFold` on `patient_id`** — a naive random split leaks visits from the same patient across train/validation and biases the local score.
- **Feature engineering can exploit a patient's other visits** — per-patient statistics, progression trend, and neighboring-visit values, all computed independently for train and test.

## Approach

**Feature engineering**
- Per-patient stats on `off`/`on` (mean/std/min/max/median/CV) — `off` correlates at 0.885 with the target but is missing ~42% of the time; imputing with the patient's own other visits instead of a global median preserves most of that signal
- Linear interpolation of `off`/`on` across visits (chronological, per patient) — better than flat patient-mean imputation
- Progression slope per patient (`off_slope`/`on_slope`/`ledd_slope`, linear fit vs. age)
- Second-order neighbors (`off_prev2`/`off_next2`), inter-visit timing gaps, local deltas (direction of change)
- Slope projection, baseline/last value per patient, intra-cohort rank features
- Standard ratios/interactions (dose per year, on/off ratio, age × disease duration, score × LEDD)

**Models**: LightGBM, XGBoost, CatBoost with early stopping (4,000 iterations max), seed averaging over 3 seeds (42/1/7), intra-fold target encoding for `cohort` and `gene` (no leakage). Final combination via Ridge stacking on OOF predictions.

## Results

| Version | Validation | RMSE |
|---|---|---|
| Linear Regression | GroupKFold (patient_id) | 4.85 |
| Random Forest | GroupKFold (patient_id) | 4.19 |
| XGBoost (original feature set) | naive split (biased) | 7.37 |
| + patient-level stats & trajectory features | GroupKFold | 4.35 |
| + progression slope, deviation from baseline | GroupKFold | 4.04 |
| + previous/next visit value | GroupKFold | 3.93 |
| Ensemble (LightGBM + XGBoost + CatBoost) | GroupKFold | 3.91 |
| + interpolation, order-2 neighbors, local deltas | GroupKFold | 3.77 |
| **+ early stopping, seed avg, target encoding, stacking** | GroupKFold | **3.57** |

Leaderboard: 57 → 31 → **19**/175

Feature engineering is the main driver of improvement — hyperparameter tuning (Optuna) brought negligible gains (~4.06 vs ~4.04) compared to richer longitudinal features.

