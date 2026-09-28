# Stock ML Pipeline — From Data to Decisions

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

An interactive, end-to-end machine learning pipeline for stock market analysis
and prediction. Load data from Yahoo Finance or your own CSV/Excel files, walk
through a guided 7-step pipeline — load, preprocess, engineer features, split,
train, evaluate, visualize — and explore predictions with interactive charts.

## ✨ Features

- **Flexible data input** — fetch real-time data from Yahoo Finance (with retry logic) or upload CSV/Excel
- **Smart preprocessing** — missing-value detection; imputation with training-set means after the split
- **Feature engineering** — moving averages, feature selection, correlation heatmap
- **Leakage-free pipeline** — imputation and standardization are fit on the training set only
- **Multiple models** — Linear Regression, Logistic Regression, K-Nearest Neighbors (auto regressor/classifier)
- **Honest evaluation** — RMSE/R² for regressors; accuracy, F1, and confusion matrix for classifiers
- **Rich visualizations** — feature importance, time series with predictions, model comparison
- **4 UI themes** — including Cyberpunk and Oceanic Blue, with animated tickers

## 🛠️ Tech stack

- **App**: Streamlit
- **ML**: scikit-learn (regression, classification, preprocessing)
- **Data**: pandas, numpy, yfinance, tenacity (retries), openpyxl (Excel support)
- **Charts**: Plotly

## ⚙️ Getting started

### Prerequisites

- Python 3.10+ and pip

### Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

## 📂 Project structure

```
├── app.py                 # The full 7-step Streamlit pipeline
├── requirements.txt
├── assets/backgrounds/    # Theme background images
└── README.md
```

## 🔌 Pipeline steps

| Step | What happens |
|---|---|
| 1. Load Data | Upload CSV/Excel or fetch from Yahoo Finance |
| 2. Preprocessing | Missing-value inspection (imputation deferred to post-split) |
| 3. Feature Engineering | Moving averages, target/feature selection, correlation matrix |
| 4. Train/Test Split | Configurable test size; imputation + scaling fit on train only |
| 5. Model Training | Linear / Logistic regression or KNN, with model-type guardrails |
| 6. Evaluation | RMSE/R² (regression) or accuracy/F1/confusion matrix (classification) |
| 7. Visualization | Feature importance, time series, model comparison, predictions |

## 📜 License

MIT — see [LICENSE](LICENSE).

---

**Author:** Muhammad Usman · FAST NUCES, Islamabad — BS Financial Technology (FinTech)
