Description of Project:
--This project aims to forecast hourly energy consumption using machine learning regression models. 
--It includes a complete end-to-end pipeline covering exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and evaluation. 
--The dataset used in this project is DAYTON_hourly.csv.

Project Goals:
--Load and prepare the dataset
--Perform Exploratory Data Analysis (EDA)
--Engineer features for time-based and categorical analysis
--Build and evaluate multiple regression models
--Optimize model performance using hyperparameter tuning
--Predict missing target values for test data
--Export results to a submission file

Fundamental Libraries & Machine Learning Models:
--Python Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, LightGBM, XGBoost, CatBoost
--Machine Learning Models: Linear Regression, K-Nearest Neighbors, Decision Trees, Random Forest, XGBoost, LightGBM

Key Features:
EDA:
---Categorical and numerical variable inspection
---Target variable distribution analysis
---Correlation matrix and heatmap
Feature Engineering:
---DateTime decomposition (year, month, day, hour)
---Outlier detection and suppression
---Rare category encoding
Modeling:
---Train-test split
---Cross-validation with RMSE metric
---Model comparison
---Hyperparameter tuning (LightGBM via GridSearchCV)
---Feature importance visualization
Prediction:
---Missing values in the target column (DAYTON_MW) are predicted using the trained LightGBM model
---Predictions are saved to DAYTON_MW_Predictions1.csv

How to Run
Install dependencies: pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost
Place the dataset: Ensure DAYTON_hourly.csv is in the same directory as the script.
Run the script:python energy_consumption_irfan_sogut.py
Output:
-RMSE scores for different models
-A .csv file (DAYTON_MW_Predictions1.csv) with the predicted values for missing target entries

Project Structure:
energy_consumption_irfan_sogut.py
DAYTON_hourly.csv
DAYTON_MW_Predictions1.csv
README.md
