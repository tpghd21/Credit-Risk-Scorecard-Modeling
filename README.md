# German Credit Risk Modeling: Scorecard & ML Benchmark

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-Benchmark-yellow) ![Optuna](https://img.shields.io/badge/Optuna-Tuning-red) ![Status](https://img.shields.io/badge/Status-Completed-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

##  Executive Summary

This project implements a comprehensive **Credit Risk Modeling** framework using the German Credit Dataset. The goal was to build a regulatory-compliant **Credit Scorecard** (based on Logistic Regression) while benchmarking it against a modern machine learning model (**XGBoost**).

A key finding of this project was that **the traditional Logistic Regression model outperformed the complex XGBoost model.** This result highlights that for smaller, structured datasets (1,000 observations), robust linear models often provide better generalization than complex ensemble methods.

**Key Achievements:**
* **Methodological Comparison:** Proved that the WOE-based Logistic Regression model offered superior performance and stability compared to XGBoost for this specific dataset.
* **Advanced Feature Engineering:** Applied **Weight of Evidence (WOE)** binning to handle non-linearity and missing values.
* **Strategic Feature Selection:** Manually excluded variables with suspiciously high Information Value (IV) to prevent data leakage.
* **Optimization:** Used **Optuna** to ensure the XGBoost benchmark was tuned to its maximum potential before comparison.

---

## Repository Structure

The analysis is divided into two main notebooks to separate business understanding from technical modeling.

| File Name | Description | Key Techniques |
|:---:|:---|:---|
| **1. [Credit_Fundamentals.ipynb](./Credit_Fundamentals.ipynb)** | **Risk Foundations & EDA**<br>Business domain analysis and deep exploratory data analysis. | • 3 Pillars (PD, LGD, EAD)<br>• Risk Segmentation<br>• Data Preprocessing |
| **2. [PD_Modelingg.ipynb](./PD_Modelingg.ipynb)** | **PD Modeling & Comparison**<br>Developing the Scorecard and benchmarking against XGBoost. | • WOE & IV Transformation<br>• **Logistic Regression (Champion)**<br>• **XGBoost (Benchmark)**<br>• **Optuna Hyperparameter Tuning** |

---

## Data Overview

* **Dataset:** German Credit Dataset (UCI Machine Learning Repository)
* **Target:** `default` (0 = Good/Repaid, 1 = Bad/Defaulted)
* **Features:**
    * *Demographics:* Age, Job, Housing, etc.
    * *Financials:* Credit amount, Duration, Checking/Savings status.

---

##  Methodology

### 1. Feature Engineering (WOE & IV)
Standard banking practice applied to both categorical and numerical features:
* **WOE (Weight of Evidence):** Transforms features to normalize relationships with the target variable and handle missing values logically.
* **IV (Information Value):** Used to rank feature importance.

### 2. Feature Selection (Crucial Step)
* **IV Filter:** Variables with weak predictive power ($IV < 0.02$) were dropped.
* **Manual Intervention:** **One variable with an exceptionally high IV (> 0.5) was manually removed.** While mathematically predictive, such variables often indicate data leakage or overfitting, so it was excluded to ensure realistic predictions.

### 3. Model Comparison
This project tested two distinct approaches:

####  Model A: Credit Scorecard (Logistic Regression) - *Selected*
* **Approach:** Built on WOE-transformed variables.
* **Pros:** Highly interpretable, compliant with Basel II/III regulations, and less prone to overfitting on small datasets.
* **Scaling:** Converted log-odds to a scorecard (Base Points: 600, PDO: 50).

####  Model B: Machine Learning (XGBoost) - *Benchmark*
* **Approach:** Gradient Boosting with **Optuna** for hyperparameter tuning.
* **Result:** Despite extensive tuning, XGBoost struggled to outperform the linear model on the Test set.
* **Insight:** Given the dataset size (1,000 rows), the complex model likely suffered from noise overfitting compared to the simpler, more robust Logistic Regression.

---

## Key Results

* **Final Model Choice:** **Logistic Regression (Scorecard)**
* **Performance:** The Scorecard model demonstrated better AUC stability and separation between "Good" and "Bad" customers compared to XGBoost.
* **Conclusion:** This project validates that for limited credit data, traditional statistical techniques (WOE + Logistic Regression) remain the gold standard for both performance and interpretability.

---

## Tech Stack

* **Language:** Python 3.x
* **Core Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **Modeling & ML:**
    * `Scikit-learn` (Logistic Regression, Metrics)
    * `XGBoost` (Gradient Boosting)
    * `Optuna` (Hyperparameter Optimization)
    * `Scorecardpy` (WOE Binning & Scaling)

---

### License
This project is open-sourced under the MIT License.
