
# Temporal Fraud Detection

A machine learning project developed for the **REACT 2026 Datathon**, organized by IEEE SEU Student Branch.

## About the Project

Our team worked on detecting fraudulent transactions by analyzing historical transaction behaviour. Instead of evaluating transactions individually, we focused on identifying unusual activities compared to the previous behaviour of customers, devices, merchants & locations.

The project uses temporal feature engineering and an ensemble of gradient-boosting models to identify potentially fraudulent transactions while avoiding future-data leakage.

## Methodology

### 1. Data Processing and Feature Engineering

We developed **95 behavioural and temporal features** from the transaction history.

Some of the key features include:

- **Customer Behaviour:** Historical transaction counts, average amounts, standard deviations, z-scores, previous transaction amounts and deviations from historical maximum values.
- **Novelty Detection:** Identifying first-time usage of devices, merchants, locations, categories and payment methods for individual customers.
- **Cross-Entity Relationships:** Tracking shared devices, customers per device, devices per customer and other entity-level relationships.
- **Transaction Velocity:** Counting transactions within 1-hour, 6-hour, 24-hour, 7-day and 30-day windows, along with burst ratios.
- **Movement Patterns:** Detecting changes in device and location, including rapid location changes between consecutive transactions.
- **Transaction Characteristics:** Amount-based features, transaction time, day of the week and other numerical indicators.

We also included historical amount statistics for devices, merchants and locations to capture unusual transaction amounts relative to those entities.

### 2. Temporal Leakage Prevention

Since the dataset follows a chronological structure, preventing future information from influencing earlier predictions was an important part of the implementation.

All features were constructed using strictly backward-looking operations, including:

- `groupby().cumcount()`
- `groupby().cumsum()` with current-row exclusion
- `groupby().shift(1)`
- `numpy.searchsorted()`

Train and test transactions were combined and sorted by timestamp before feature construction. This allowed earlier transactions from the test period to contribute to the historical behaviour of later transactions without using any future information.

### 3. Validation Strategy

Instead of random K-fold validation, we used a chronological holdout to better reflect the actual prediction scenario.

- Training: 672,477 transactions (before July 1, 2026)
- Validation: 59,465 transactions (July 1–15, 2026)
- Fraud cases in validation: 926

The evaluation metric was **PR-AUC (Average Precision)**.

### 4. Model Development and Ensemble

We experimented with three gradient-boosting algorithms:

| Model | Holdout PR-AUC |
|---|---:|
| LightGBM | 0.52106 |
| CatBoost | 0.51973 |
| HistGradientBoosting | 0.51676 |
| **Final Rank Ensemble** | **0.52164** |

The final predictions were generated using three seeds for each model family. We combined their predictions through equal-weight rank averaging.

Different blending weights were also evaluated before retaining the equal-weight combination.

## Results

| Evaluation | PR-AUC |
|---|---:|
| Holdout Validation | 0.52164 |
| Public Leaderboard | 0.54734 |
| Private Leaderboard | 0.54609 |

Our team was selected among the 15 teams invited to the final round following notebook evaluation, where we presented our approach and findings.

## Technologies Used

Python · Pandas · NumPy · Scikit-learn · LightGBM · CatBoost · SciPy

## Repository Contents

- `Temporal-Fraud-Detection.ipynb` — Complete implementation, including feature engineering, validation, model training, ensemble development, and submission generation.
- `Temporal Fraud Detection Report.pdf` — Detailed technical report covering the methodology, experiments, results and limitations.
