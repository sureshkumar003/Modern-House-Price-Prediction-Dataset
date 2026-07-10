#  Comprehensive Real Estate Pricing Valuation Framework Using Supervised

**Student:** Sureshkumar Karuppusamy  
**Matriculation Number:** 51134495  
**Class:** MSc Data Science 120B  
**Dataset:** [Modern House Price Prediction Dataset](https://www.kaggle.com/datasets/miadul/house-price-prediction-
dataset)

---

## Project Overview

The real estate market plays a critical role in macroeconomic stability and individual wealth
accumulation. Accurate property valuation is essential for stakeholders, including buyers, sellers, institutional
lenders, and urban planners. Traditionally, valuations relied heavily on manual appraisals, which are prone to
subjectivity, time-consuming, and struggle to scale across massive spatial regions. The introduction of
supervised machine learning tools has completely transformed this domain by enabling automated, objective,
and multi-dimensional valuation boundaries over comprehensive data assets.

---

## Problem Statement:
Accurately forecasting real estate prices is crucial for home buyers, investors, and urban planners. Housing prices are influenced by complex, interacting socioeconomic and geographical factors (such as regional income levels and proximity to employment hubs). This project aims to build a predictive machine learning model that leverages district-level census data to accurately estimate median home values, thereby minimizing financial risk in real estate investments.

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — miadul |
| Rows | 50,000 |
| Columns | 18 |
| Target Variable | `Continuous Numerical Variable` |
| Task Type | Regression |

---

## Repository Structure

```
ML assignment /
├── RQ1.ipynb       # Baseline models: Linear Reg, Decision Tree, k-NN
├── RQ2.ipynb       # Advanced models: Random Forest, XGBoost, SVR
├── RQ3.ipynb       # Impact of preprocessing strategies
├── RQ4.ipynb       # Feature importance + SHAP analysis
├── RQ5.ipynb       # Model rankings across different metrics
├── RQ6.ipynb       # Cross-validation and noise/missingness tests
├── RQ7.ipynb       # Final decision matrix and recommendation
├── README.md
└── requirements.txt
```

Each notebook is **self-contained** — it reads the raw dataset and outputs figures (PDF) and tables (CSV) independently.

---

## Research Questions

| # | Research Question | Output |
|---|---|---|
| RQ1 | What is the correlation between median income (MedInc) and the median house value in a district? | 
| RQ2 | Does the average number of rooms per household (AvgRooms) have a linear relationship with housing prices? |
| RQ3 | How drastically do geographical coordinates (Latitude and Longitude) impact housing prices in California?
| RQ4 | Does treating population density as a feature improve model accuracy? |
| RQ5 | Which machine learning model achieves the lowest Root Mean Squared Error (RMSE) on this dataset? |
| RQ6 | Is ensemble learning (Random Forest) significantly better than a baseline Linear Regression model for this task? |

---

## Models Used

- Linear Regression
- Decision Tree
- k-Nearest Neighbors (k-NN)
- Random Forest
- XGBoost
- Support Vector Regression (SVR)
- Evaluation Metrics
  
---

## Expected Outputs

Each notebook saves its results automatically:

- `figures/` — Publication-ready figures saved as PDF
- `tables/` — Result tables saved as CSV

