# A Data-Driven Framework for Product Selection in E-Commerce

**Using NLP and Machine Learning to Identify High-Demand, Low-Competition Niches on Amazon**

MSc Data Analytics Dissertation
Student: Syed Aqeel (24060071)
Supervisor: Dr. Subeksha
London Metropolitan University

---

## Overview

This project develops and validates a composite scoring framework that helps
independent e-commerce sellers identify profitable product niches on Amazon.
It combines Natural Language Processing (NLP) of customer reviews with machine
learning classification to score product niches across four dimensions: demand,
competition, pricing stability, and sentiment gap.

## Research Questions

- **RQ1** — To what extent can NLP-based review mining reliably identify unmet
  customer needs and market gaps in e-commerce product categories?
- **RQ2** — Which machine learning model most accurately classifies e-commerce
  product niches by profitability potential?
- **RQ3** — How effectively does a composite scoring framework — combining
  demand, competition, pricing, and sentiment signals — predict real-world
  product performance?

## Methodology

The project follows the **CRISP-DM** framework.

**Multi-stage sentiment analysis:**
- VADER — lexicon-based baseline
- BERT / RoBERTa — deep contextual sentiment (via HuggingFace Transformers)
- PyABSA — aspect-based sentiment extraction

**Machine learning models:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LSTM (demand pattern modelling)

**Composite scoring weights:**
| Dimension          | Weight |
|--------------------|--------|
| Demand             | 30%    |
| Competition        | 25%    |
| Pricing stability  | 20%    |
| Sentiment gap      | 25%    |

## Folder Structure

```
ecommerce-product-selection/
├── data/
│   ├── raw/            Original untouched datasets
│   ├── processed/      Cleaned, preprocessed data
│   └── external/       Google Trends, supplementary data
├── notebooks/          Jupyter/Colab notebooks (numbered by stage)
├── src/                Reusable Python modules
├── models/             Saved trained models
├── outputs/
│   ├── figures/        Charts and visualisations (300 dpi)
│   └── results/        Model results, scores, tables
├── docs/               Dissertation drafts and notes
├── requirements.txt    Python dependencies
└── README.md           This file
```

## Data Sources

- **Amazon reviews** — McAuley Amazon Reviews 2023 dataset (HuggingFace Hub),
  Home & Kitchen category. Cited as Ni, Li & McAuley (2019), EMNLP-IJCNLP.
- **Google Trends** — accessed via `pytrends` for contemporary demand signals.

## Data Notes

A 100,000-review working sample was extracted from the Home & Kitchen category.
Following supervisor guidance, the `verified_purchase` field was analysed to
assess whether unverified reviews introduce bias. Verified and unverified
reviews were found to differ substantially in length (Cohen's d = 0.93) and in
the proportion of one-star ratings (6.62% versus 2.55%). Verified reviews were
therefore adopted as the primary corpus, with unverified reviews retained as a
comparison set rather than discarded.

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/aqeel-gardezi/MSc-Project.git
cd MSc-Project

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

Transformer-heavy work (BERT, RoBERTa, PyABSA) is run in Google Colab to make
use of free GPU acceleration.

## Notebooks

| Notebook                      | Purpose                                     |
|-------------------------------|---------------------------------------------|
| 01_data_collection.ipynb      | Data collection, sampling, verified analysis |
| 02_preprocessing.ipynb        | Clean and prepare the corpus                |
| 03_sentiment_analysis.ipynb   | VADER, BERT/RoBERTa, PyABSA                 |
| 04_ml_models.ipynb            | Train and compare ML models                 |
| 05_composite_scoring.ipynb    | Build and validate scoring framework        |

## Ethics

This project uses only publicly available datasets and APIs in accordance with
platform terms of service. No personal data is collected or stored.

## References

- Devlin et al. (2019) — BERT
- Liu et al. (2019) — RoBERTa
- Wolf et al. (2020) — HuggingFace Transformers
- Hutto & Gilbert (2014) — VADER
- Yang & Li (2023) — PyABSA
- Ni, Li & McAuley (2019) — Amazon Reviews dataset
