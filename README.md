# German Credit Risk Modeling: Scorecard vs XGBoost


## Executive Summary

This project implements a comprehensive **Credit Risk Modeling** framework using the German Credit Dataset. The primary objective was to build a regulatory-compliant **Credit Scorecard** (based on Logistic Regression) while benchmarking it against a modern machine learning model (**XGBoost**).

A key finding of this project was that the traditional Logistic Regression model outperformed the complex XGBoost model. This is not merely due to the small dataset size, but primarily because **WOE (Weight of Evidence) Binning successfully transformed non-linear features into monotonic trends**, allowing the linear model to capture the signal effectively and robustly.

**Key Achievements:**
* **Strategic Feature Engineering:** Applied WOE Binning to ensure **monotonic relationships** between independent variables and the default rate, maximizing the performance of the linear model.
* **Scorecard Development:** Converted model probability outputs into a transparent point-based system.
* **Rigorous Selection:** Manually excluded variables with suspiciously high Information Value (IV) to prevent data leakage.
* **Model Comparison:** Proved that a well-engineered linear model can outperform complex ensemble methods in credit scoring contexts.

---

## Repository Structure

The analysis is divided into two main notebooks to separate business understanding from technical modeling.

| File Name | Description | Key Techniques |
|:---|:---|:---|
| **1. Credit_Fundamentals.ipynb** | **Risk Foundations & EDA**<br>Business domain analysis and deep exploratory data analysis. | • 3 Pillars (PD, LGD, EAD)<br>• Risk Segmentation<br>• Data Preprocessing |
| **2. PD_Modelingg.ipynb** | **PD Modeling & Comparison**<br>Developing the Scorecard and benchmarking against XGBoost. | • WOE & IV Transformation<br>• **Logistic Regression (Champion)**<br>• **XGBoost (Benchmark)**<br>• **Optuna Hyperparameter Tuning** |

---

## Methodology

### 1. Feature Engineering (WOE & Monotonicity)
The core strength of this project lies in the preprocessing stage using **Weight of Evidence (WOE)**.
* **Monotonic Trends:** Instead of feeding raw data into the model, variables were binned (grouped). We ensured that the WOE values across these bins showed a **monotonic trend** (either strictly increasing or decreasing) with respect to the default rate.
* **Linearization:** This process effectively "linearized" non-linear relationships, making them perfectly suitable for Logistic Regression.

### 2. Feature Selection
* **IV Filter:** Variables with weak predictive power ($IV < 0.02$) were dropped.
* **Manual Intervention:** One variable with an exceptionally high IV (> 0.5) was manually removed to prevent data leakage and overfitting.

### 3. Scorecard Mechanics & Example
The final output of the Logistic Regression (Log-odds) was scaled into a user-friendly score.
* **Scaling Parameters:**
    * Base Points: 600
    * Odds: 1:19
    * PDO (Points to Double the Odds): 50

#### How the Score is Calculated (Example)
The final credit score is the sum of the **Base Score** and the points assigned to each attribute.

| Attribute | Value (Customer A) | Points Assigned |
|:---|:---|:---:|
| **Base Score** | - | **600** |
| Account Status | No Checking Account | +55 |
| Duration | 12 Months (Short Term) | +20 |
| Credit History | Existing Credits Paid Back | +10 |
| Purpose | New Car | -15 |
| ... | ... | ... |
| **Final Credit Score** | | **670** |

*Interpretation: A score of 670 places Customer A in the "Low Risk" tier, leading to automatic loan approval.*

---

## Model Comparison Results

### Champion Model: Logistic Regression (Scorecard)
* **Performance:** Superior stability and generalization on the test set.
* **Why it worked:** The success of this model confirms that **forcing monotonicity via WOE binning** creates a highly robust predictor for credit risk data, often surpassing complex non-linear models which may overfit noise in smaller datasets (1,000 observations).

### Benchmark Model: XGBoost
* **Approach:** Gradient Boosting optimized with **Optuna** for hyperparameter tuning.
* **Result:** Despite extensive tuning, XGBoost struggled to significantly outperform the linear model. This validates the industry standard of preferring interpretable linear models when strong feature engineering (WOE) is applied.

---

## Tech Stack

* **Language:** Python 3.x
* **Core Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **Modeling & ML:**
    * `Scikit-learn` (Logistic Regression)
    * `XGBoost` (Gradient Boosting)
    * `Optuna` (Hyperparameter Optimization)
    * `Scorecardpy` (WOE Binning & Scaling)

---


### License
This project is open-sourced under the MIT License.
