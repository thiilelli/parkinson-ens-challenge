# Parkinson Motor Score Prediction — ENS Data Challenge

Predicting the true OFF motor score for Parkinson's disease patients.

## Approach

Three models compared on tabular clinical data:
- Baseline: Linear Regression
- Random Forest (n_estimators=100)
- XGBoost tuned (n_estimators=500, max_depth=4, lr=0.05)

## Feature Engineering

- time_since_diagnosis = age - age_at_diagnosis
- dose_per_year = ledd / (time_since_diagnosis + 1)
- on_off_ratio = on / (off + 1)
- age_diagnosis_interaction = age * time_since_diagnosis
- cohort statistics (mean ON/OFF scores, mean LEDD)
- missing value indicator for time_since_intake_off

## Results

## Results

| Model | RMSE |
|---|---|
| Linear Regression | 9.67 |
| Random Forest | 7.41 |
| XGBoost (default, CV 5-fold) | 7.51 ± 0.10 |
| XGBoost (tuned, CV 5-fold) | **7.37 ± 0.11** |

## Setup

pip install -r requirements.txt

Open notebooks/p_analysis.ipynb in Jupyter.

## Data

ENS Data Challenge #159 — clinical data from Parkinson's disease patients.
Features include medication dosage, ON/OFF motor scores, cohort information.

## License

MIT
