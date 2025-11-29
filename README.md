# Algorithmic Trading Forensics — Manipulation Detection

**Bank Nifty high-frequency forensics (SEBI-style patch analysis & anomaly detection)**
*October 2025* — Notebook: `jane.ipynb`

---

## Overview

This repository contains the end-to-end analysis, feature engineering, modeling and visualization pipeline used to investigate suspected intraday market-manipulation events in **Bank Nifty** using **1-minute tick data (≈850k+ observations)**.

The analysis reproduces SEBI-style “Patch I / Patch II” forensic checks, engineers microstructure features (e.g., **Jump_Open**, **Reversal_Ratio**), and applies econometric (GARCH), supervised (Logistic Regression) and unsupervised (Isolation Forest) methods to surface anomalous days and potential manipulation patterns.

**Primary goals**

* Reconstruct SEBI intra-day patches and metrics.
* Engineer interpretable features that capture the “pump then dump” (two-patch) signature.
* Use a mixture of statistical and ML approaches to rank and flag anomalous days.
* Produce clear visual evidence (plots) for manual forensic review.

---

## Key findings (summary)

* Feature engineering produced interpretable signals such as **Jump_Open**, **Patch1_Change**, **Patch2_Change**, and **Reversal_Ratio** that capture the patch dynamics.
* A Logistic Regression trained on SEBI-labeled days can achieve **100% recall** on known manipulated days in the reported split, but precision is low because many additional days are flagged (false positives).
  **Important:** high recall doesn't prove manipulation on all flagged days — it shows the model recognizes the SEBI-labeled pattern and flags other days that resemble it.
* An **Isolation Forest** trained on the same features ranks anomalous days; top-ranked days are the most structurally different from typical market days according to the chosen features.
* A GARCH(1,1) residual-analysis flags days with statistically extreme standardized residuals — an orthogonal signal of anomalous volatility behavior.

**Caveat:** `Y=1` are SEBI-identified days; `Y=0` is assumed normal but may contain unlabeled anomalies. Treat model outputs as forensic leads (to be investigated), not definitive proof.

---

## Repo structure (suggested)

```
.
├── data/
│   └── bank-nifty-1m-data.csv          # (NOT included) raw 1-min tick CSV
├── notebooks/
│   └── jane.ipynb                      # primary Colab notebook (this project)
├── src/
│   ├── features.py                     # feature engineering helpers (optional)
│   ├── models.py                       # model training & scoring helpers (optional)
│   └── viz.py                          # plotting helpers (optional)
├── outputs/
│   ├── figures/                        # generated PNG/PDF plots
│   └── rankings/                       # anomaly ranking CSVs
├── README.md
└── requirements.txt
```

---

## Files included

* `notebooks/jane.ipynb` — main Colab notebook which:

  * loads and parses the data,
  * constructs datetime indexes,
  * defines SEBI manipulation dates,
  * computes intra-day patches and features,
  * runs Logistic Regression (supervised),
  * runs GARCH analysis,
  * runs Isolation Forest (unsupervised),
  * generates visualizations and ranked anomaly lists.

> `data/bank-nifty-1m-data.csv` is **not included** (privacy / licensing). Users must provide equivalent tick-level data.

---

## Data requirements & format

**Input:** a CSV (`bank-nifty-1m-data.csv`) with at minimum the columns (case-insensitive):
`Date`, `Time`, `Open`, `High`, `Low`, `Close`, `Volume` (optional but useful).

* Expected date format in notebook: `DD-MM-YYYY`
* Expected time format: `HH:MM:SS`
* Should cover the sample period (e.g., `2023-01-01` → `2024-03-22`) and include full trading minutes (09:15 → 15:30).

If your CSV uses different headers or formats, edit the parsing code:

```python
df.columns = df.columns.str.lower()
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
```

---

## Patch definitions

* **Patch I (Options Build / Pump):** `09:15` — `11:47`
* **Patch II (Monetization / Dump):** `11:49` — `15:30`
* Reversal boundary: around `11:47`–`11:49`

### Engineered features

* **Jump_Open**: max high in `09:15`–`09:30` relative to open.
* **Patch1_Change**: Patch I close − patch1 open.
* **Patch2_Change**: Patch II close − patch2 open.
* **Reversal_Ratio**: `- (Patch2_Change / Patch1_Change)`.

---

## Models & analytics implemented

1. **Descriptive visualizations:** full-day plots, patch overlays, normalized SEBI-day comparison.
2. **Econometric:** GARCH(1,1) on log-returns → standardized residuals → anomaly flags.
3. **Supervised:** Logistic Regression with `class_weight='balanced'`

   * High recall on SEBI days.
   * Many additional flagged days → potential leads.
4. **Unsupervised:** Isolation Forest on normalized features → ranked anomaly list (top 20 days visualized).

---

## How to run (quick start)

### 1. Clone the repo

```bash
git clone <repo-url>
cd <repo-directory>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt
