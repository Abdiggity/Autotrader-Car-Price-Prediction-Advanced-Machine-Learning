#  Used Car Price Prediction – Advanced Machine Learning (Autotrader UK)

This repository showcases an **advanced machine learning pipeline** for predicting **used car prices** using real-world data from **Autotrader UK**.  
The project integrates **feature engineering**, **ensemble regression**, and **model interpretability techniques** to build a transparent, high-performing predictive model.

---

##  Project Overview

Modern automotive marketplaces like Autotrader host millions of listings, providing valuable data for understanding how attributes such as **mileage**, **engine size**, **brand**, and **age** influence resale prices.  

This project applies **state-of-the-art supervised learning** techniques to:
- Predict car prices accurately using tabular data.
- Engineer domain-specific features for vehicle valuation.
- Interpret model behaviour using explainable AI tools such as **SHAP** and **Partial Dependence Plots**.
- Compare ensemble learners (Random Forests, Gradient Boosting) with classical methods.

---

##  Methodology

### 1. Data Preprocessing
- **Source:** Cleaned dataset derived from *Autotrader UK listings* (not publicly shared for licensing reasons).
- Removed outliers and extreme prices (> £100,000).
- Handled missing data with imputation and scaling.
- Encoded categorical variables using **target encoding** and **label encoding**.

### 2. Feature Engineering
- Created **Vehicle Age** and **Mileage-per-Brand Feature (MMBF_target_encoded)** to capture depreciation effects.
- Aggregated and normalized numerical predictors.
- Generated polynomial interaction terms for regression benchmarking.

### 3. Model Development
Models trained and compared:
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Polynomial Regression (baseline)**
- **Stacked Ensemble** (meta-learner combining top models)
- **Weighted Average Ensemble**

### 4. Model Evaluation
Metrics used:
| Metric | Description | Example (Best Model) |
|---------|--------------|---------------------|
| MAE | Mean Absolute Error | **2,216** |
| RMSE | Root Mean Squared Error | **3,968** |
| R² | Coefficient of Determination | **0.93** |

All models validated using **k-fold cross-validation** to ensure robustness.

### 5. Model Interpretability
- **SHAP (SHapley Additive exPlanations):** Assessed global and local feature importance.
- **Partial Dependence Plots (PDP):** Visualized feature effects on predicted prices.
- **Feature Elimination (RFE):** Ranked variable importance to simplify model complexity.

### 6. Dimensionality Reduction & Clustering
- Applied **PCA** and **Isomap** to visualize latent relationships.
- Tested **K-Means** and **Agglomerative Clustering** for unsupervised feature grouping (not improving RMSE but explored for insight).

---

##  Key Insights

- The **MMBF target-encoded feature** was the single strongest predictor of price, outperforming raw brand or mileage inputs.  
- **Vehicle Age**, **Engine Size**, and **Mileage** showed clear, interpretable nonlinear relationships with price.  
- Ensemble methods generalized better than single models, especially for mid-priced vehicles.  
- SHAP analysis confirmed that the model aligned with real-world automotive valuation logic.  
- PCA and clustering analyses revealed brand-specific price clusters (e.g., luxury vs economy segments).

---

##  Results Summary

| Model | MAE | RMSE | R² | Notes |
|--------|------|------|----|-------|
| Random Forest (tuned) | 2,216 | 3,968 | 0.93 | Best standalone performer |
| Gradient Boosting | 2,289 | 4,034 | 0.928 | Slightly less robust |
| Stacked Ensemble | 2,234 | 3,957 | **0.931** | Best overall performance |
| Polynomial Regression | 3,312 | 5,892 | 0.78 | Nonlinear baseline comparison |

---

## Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.10 |
| **Data Processing** | pandas, NumPy, scikit-learn |
| **Modeling** | RandomForestRegressor, GradientBoostingRegressor, StackingRegressor |
| **Interpretability** | SHAP, PDPBox |
| **Visualization** | Matplotlib, Seaborn |
| **Dimensionality Reduction** | PCA, Isomap |
| **Notebook Environment** | Jupyter Notebook / Google Colab |

---

.
├── README.md                         # Project overview & methodology
├── Adv_ML_notebook_sub.ipynb         # Main analysis notebook
├── 18062413_ADV_ML_FULL_REPORT.pdf   # Detailed written report
└── (data excluded)                   # Autotrader data not included due to license


