# 🧾 Customer Churn Prediction

An end‑to‑end **machine learning** project to predict **customer churn** for subscription/contract‑based businesses.  
This repository includes data preprocessing, feature engineering, model training (Logistic Regression, Random Forest, XGBoost), **threshold tuning** for business goals, and rigorous evaluation (ROC‑AUC, PR‑AUC, F1).

---

## 📖 Overview
Churn prediction helps allocate retention efforts efficiently (discounts, outreach, win‑back campaigns).  
This project builds a reproducible pipeline and compares multiple models, focusing on **recall** for early churn capture while controlling precision via **threshold selection**.

**Highlights**
- Clean pipelines with `ColumnTransformer` + `Pipeline` (no leakage)  
- Robust evaluation on **stratified** train/test split + optional cross‑validation  
- **Calibration** for reliable probabilities (isotonic)  
- Cost‑aware thresholding (maximize **F2** or Youden‑J)  
- Feature importance (model‑specific and **SHAP** optional)

---

## 🗂️ Dataset
- **Target**: `churn` (1 = churned, 0 = retained)  
- **Typical features**: demographics, contract type, tenure, monthly charges, payment method, product usage (sessions, tickets, failures), engagement scores.  
- Example CSVs:
```
data/
├─ train.csv
└─ test.csv
```
> Replace with your own data; ensure consistent column names between train/test.

---

## 🔧 Preprocessing & Feature Engineering
- Missing value imputation (numeric/ categorical)  
- Scaling numerical features (**fit on train only**)  
- One‑hot encoding for categoricals (ignore unknowns at inference)  
- Domain features (e.g., ARPU, tenure buckets, complaints in last 30d)  
- Optional: interaction terms and target‑encoding (with CV to avoid leakage)

---

## 🧠 Models
- **Logistic Regression** (baseline + interpretability)  
- **Random Forest** (non‑linear baseline)  
- **XGBoost** (strong tabular learner)  
- *(Optional)* LightGBM, CatBoost

---

## 📈 Evaluation
Primary metrics:
- **ROC‑AUC** — ranking ability
- **PR‑AUC** — useful under class imbalance
- **F1 / F2** — balance or recall‑heavy scenarios
- **Confusion matrix** at chosen threshold

**Threshold selection**:
- Sweep thresholds to maximize **F2** (recall‑oriented) or **Youden‑J**  
- Choose operating point aligned with business cost matrix

---

## 🧩 Repository Structure (suggested)
```
Customer-Churn-Prediction/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_training_baselines.ipynb
│  ├─ 03_threshold_calibration.ipynb
├─ src/
│  ├─ data.py          # load/split; stratification; leakage-safe transforms
│  ├─ features.py      # preprocessors, domain features
│  ├─ models.py        # model builders (logreg/RF/XGB)
│  ├─ train.py         # fit + CV
│  ├─ eval.py          # metrics, curves, threshold sweep
│  ├─ explain.py       # SHAP (optional)
├─ reports/figures/    # ROC/PR curves, CM, feature importance
├─ data/               # (gitignored)
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚙️ Setup & Usage
1) **Clone & install**
```bash
git clone https://github.com/ziaee-mohammad/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

2) **Run notebooks** (EDA → training → threshold)
```bash
jupyter notebook
```

3) **Or use scripts (if added)**
```bash
python -m src.train --model xgboost
python -m src.eval  --threshold auto_f2
```

---

## 📦 Requirements (example)
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
shap           # optional
```

---

## ✅ Reproducibility & Anti‑Leakage Checklist
- Use **stratified** splits; fix random seeds (NumPy/sklearn)  
- Keep scaling/encoding **inside** the pipeline; `fit` on train only  
- Report ROC/PR curves; store chosen threshold and rationale  
- Calibrate probabilities if needed (`CalibratedClassifierCV`)  
- Log data windows used (train/val/test) for temporal stability

---

## 📈 Example Results (replace with your numbers)
| Model | ROC‑AUC | PR‑AUC | F1 | F2 |
|------|--------:|------:|---:|---:|
| Logistic Regression | 0.83 | 0.48 | 0.58 | 0.65 |
| Random Forest | 0.86 | 0.52 | 0.61 | 0.68 |
| **XGBoost** | **0.89** | **0.57** | **0.64** | **0.72** |

> Include confusion matrix at the selected operating threshold.

---

## 🧑‍💻 Author
**Mohammad Ziaee** — Computer Engineer | AI & Data Science  
📧 moha2012zia@gmail.com  
🔗 https://github.com/ziaee-mohammad

---

## 🏷 Tags
```
data-science
machine-learning
classification
churn-prediction
customer-analytics
tabular-data
model-evaluation
feature-engineering
python
jupyter-notebook
``

---

## 📜 License
MIT — free to use and adapt with attribution.
