Assignment Proposal

Project Title: Comprehensive Real Estate Pricing Valuation Framework Using Supervised
Regressors
Dataset Name: Modern House Price Prediction Dataset

Dataset Link: https://www.kaggle.com/datasets/miadul/house-price-prediction-
dataset

Dataset Size: 50,000 Rows × 18 Columns
Target Variable: price (Continuous Numerical Variable)
Type of Task: Supervised Learning — Regression

Project Title: Comprehensive Real Estate Pricing Valuation Framework Using Supervised
Regressors
Dataset Name: Modern House Price Prediction Dataset

Dataset Link: https://www.kaggle.com/datasets/miadul/house-price-prediction-
dataset

Dataset Size: 50,000 Rows × 18 Columns
Target Variable: price (Continuous Numerical Variable)
Type of Task: Supervised Learning — Regression

Problem Statement:
Accurately forecasting real estate prices is crucial for home buyers, investors, and urban planners. Housing prices are influenced by complex, interacting socioeconomic and geographical factors (such as regional income levels and proximity to employment hubs). This project aims to build a predictive machine learning model that leverages district-level census data to accurately estimate median home values, thereby minimizing financial risk in real estate investments.

Research Questions:
1.What is the correlation between median income (MedInc) and the median house value in a district?
2.Does the average number of rooms per household (AvgRooms) have a linear relationship with housing prices?
3.How drastically do geographical coordinates (Latitude and Longitude) impact housing prices in California?
4.Does treating population density as a feature improve model accuracy?
5.Which machine learning model achieves the lowest Root Mean Squared Error (RMSE) on this dataset?
6.Is ensemble learning (Random Forest) significantly better than a baseline Linear Regression model for this task?

Proposed Methodology:
Exploratory Data Analysis (EDA): Visualize distributions, check for missing values, and analyze feature correlations using a heatmap.
Data Preprocessing: Standardize features using StandardScaler to ensure scale-invariant models perform optimally. Split data into training (80%) and testing (20%) sets.
Model Training: Implement three distinct algorithms:
Baseline: Linear Regression
Advanced Linear: Ridge Regression (L2 Regularization)
Non-Linear Ensemble: Random Forest Regressor
Evaluation: Compare models using standard regression metrics across test data.
Evaluation Metrics:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R 
2
  Score (Coefficient of Determination)
Expected Figures and Tables:
Figure 1: Correlation Heatmap of features.
Figure 2: Scatter plot of Median Income vs. Median House Value.
Figure 3: Feature Importance Plot from the Random Forest model.
Table 1: Model Performance Comparison Matrix (Metrics across all models).
