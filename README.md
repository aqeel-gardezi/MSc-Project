# A Data-Driven Framework for Product Selection in E-Commerce

**Using NLP and Machine Learning to Identify High-Demand, Low-Competition Niches on Amazon and eBay**

MSc Data Analytics Dissertation
Student: Syed Aqeel (24060071)
Supervisor: Dr. Subeksha
London Metropolitan University

---

## Overview

This project develops and validates a composite scoring framework that helps
independent e-commerce sellers identify profitable product niches on Amazon and
eBay. It combines Natural Language Processing (NLP) of customer reviews with
machine learning classification to score product categories across four
dimensions: demand, competition, pricing stability, and sentiment gap.

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

**NLP pipeline (hybrid):**
- VADER — lexicon-based baseline sentiment
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
│   ├── figures/        Charts and visualisations
│   └── results/        Model results, scores, tables
├── docs/               Dissertation drafts and notes
├── requirements.txt    Python dependencies
└── README.md           This file
```

## Data Sources

- Amazon Reviews — Kaggle / Harvard Dataverse
- eBay listings — eBay Browse API
- Google Trends — via pytrends

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ecommerce-product-selection.git
cd ecommerce-product-selection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

For transformer-heavy work (BERT, RoBERTa, PyABSA), run the relevant notebooks
in Google Colab to make use of free GPU acceleration.

## Notebooks

| Notebook                      | Purpose                                  |
|-------------------------------|------------------------------------------|
| 01_data_collection.ipynb      | Collect data from Kaggle, eBay, Trends   |
| 02_preprocessing.ipynb        | Clean and prepare data                   |
| 03_nlp_sentiment.ipynb        | VADER, BERT/RoBERTa, PyABSA pipeline     |
| 04_ml_models.ipynb            | Train and compare ML models              |
| 05_composite_scoring.ipynb    | Build and validate scoring framework     |

## Ethics

This project uses only publicly available datasets and APIs in accordance with
platform terms of service. No personal data is collected or stored.

## References

- Devlin et al. (2019) — BERT
- Liu et al. (2019) — RoBERTa
- Wolf et al. (2020) — HuggingFace Transformers
- Hutto & Gilbert (2014) — VADER
- Yang & Li (2023) — PyABSA
