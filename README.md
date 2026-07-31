# Apartment Rental Price Prediction
# Purpose/Objective:
- To collect and prepare the apartment rental dataset for modeling by handling missing values, treating outliers, encoding categorical features, and scaling numerical features, producing a clean and reliable dataset.
- To perform exploratory data analysis (EDA) and visualisation in order to identify the key relationships and patterns between property features and rental price
- To build and compare four regression models to predict apartment rental prices from property features: 
    **Linear Regression
    K-Nearest Neighbours Regressor
    Random Forest Regressor
    Gradient Boosting Regressor**
- To evaluate and compare the performance of the four models using RMSE, MAE, and R², and to identify the most accurate and reliable model for rental price prediction.
- To deploy the best-performing model as a functional prototype that allows users to estimate apartment rental prices interactively.

Dataset Source: https://www.kaggle.com/datasets/shashanks1202/apartment-rent-data

# Problem Statement:
- Which features of the property have the greatest impact on the price of an apartment.
- How effective are machine learning regression models in prediction of the rental price from these features.
- Which of the models gives the best predictions in terms of accuracy and reliability (based on the results of **RMSE, MAE, and R²**).

# Method:
Cross Industry Standard Process for Data Mining (CRISP-DM) framework with Python & Streamlit
Business Understanding -> Data Understanding -> Data Preparation -> Modelling -> Evaluation -> Deployment

# Code:

Before implementation setup environtment:
python -m venv rent_env
rent_env\Scripts\Activate.ps1  
pip install -r requirements.txt
- [Business Understanding & Data Understanding] Includes visualization for business & data understanding of the dataset
- [Data Preparation] Includes data cleaning, data transformation and data outlier detection coding before modelling
- [Modelling] Includes all models training setup and enhancements
- [Evaluation] Includes all suitable evaluation shared by all models to do comparison
- [Deployment] Streamlit visualization and prediction
# Apartment Rental Price Prediction

BMDS2003 Data Science — CRISP-DM project.

## Repo structure

```
├── requirements.txt          # pinned dependencies — everyone installs the SAME versions
├── src/
│   ├── config.py              # single source of truth for 3.1 column decisions
│   └── data_cleaning.py       # shared 3.2 cleaning functions — import, don't copy-paste
├── visualization/
│   ├──  # visualization of dataset?
├── app/
│   └── streamlit_app.py       # compulsory deployment prototype
├── data/                      # gitignored — dataset is loaded via kagglehub, not committed
└── outputs/figures/           # save EDA/model plots here for pasting into the Google Doc report
```

## Setup (everyone runs this once)

```bash
git clone <repo-url>
cd apartment_rental_price_prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Using the same `requirements.txt` matters more than it sounds — if one teammate has scikit-learn 1.3 and another has 1.5, model results and pickled objects can silently differ or break between machines. Check the pinned versions against your own with `pip freeze` before installing over an existing environment; adjust if your installed versions differ significantly.

## Deploying the prototype

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app" → select this repo, branch `main`, file path `app/streamlit_app.py`.
4. Streamlit Cloud auto-detects `requirements.txt` at the repo root.
5. Every push to `main` auto-redeploys — no manual redeploy step needed.

Free-tier limits worth knowing: ~1 GB RAM, the app sleeps after 12 hours with no traffic (first visitor after that waits ~30s for it to wake), and you get one private app (unlimited public apps).

## Dataset

[Apartment Rent Data](https://www.kaggle.com/datasets/shashanks1202/apartment-rent-data) — loaded at runtime via `kagglehub`, not committed to the repo.
