"""
Model Validation Wrappers
===========================
Wrappers for model calibration explainability, and anomaly detection

This module combines three types of model validation:
1. Calibration - Platt Scaling for probability calibration
2. Explainability - SHAP-based model interpretation
3. Anomaly Detection - Isolation Forest for distribution monitoring
"""

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from validation_metrics import calculate_ece

# ============================================================================
# SECTION 1: CALIBRATION FUNCTIONS
# ============================================================================

def apply_sigmoid_calibration(models, X_train, y_train, cv =5, ensemble =True):
    """
    Apply Sigmoid Calibration (Platt Scaling) to models

    Parameters
    ----------
    models : dict
        Dictionary of {model_name: model_object}
    X_train : pd.DataFrame
        Training features
    y_train : np.darray
        Training labels
    cv : int, default =5
        Number of cross-validation folds
    ensemble : bool, default = True
        If True, use all CV models in ensemble
    
    Returns
    -------
    dict
        Dictionary of calibrated models {model_name: calibrated_model}
    """
    calibrated_models ={}

    for name, model in models.items():
        calibrated_clf = CalibratedClassifierCV(
            model,
            method = 'sigmoid',
            cv =cv,
            ensemble = ensemble
        )

        calibrated_clf.fit(X_train, y_train)
        calibrated_models[name] = calibrated_clf
    print("All models calibrated")
    return calibrated_models


# ============================================================================
# SECTION 2: EXPLAINABILITY FUNCTIONS (SHAP)
# ============================================================================
def compute_shap_values(calibrated_model, X_train, X_test,
                        sample_size =200, n_background =100,
                        nsamples =100, random_state =42):
    """
    Compute SHAP values for calibrated model using Kernel SHAP

    Parameters
    ----------
    calibrated_model
        Calibrated model object
    X_train : pd.DataFrame 
        Training data(for background)
    X_test : pd.DataFrame
        Test data (for SHAP computation)
    sample_size : int, default = 200
        Number of test samples to explain
    n_background : int, default = 100
        Number of background samples for KernelSHAP
    nsamples : int, default =100
        Number of samples for KernelSHAP approximation
    random_state : int, default =42

    Return
    -------
    tuple
        (explainer, shap_values, X_sample)
    """
    # Sample for SHAP computation
    np.random.seed(random_state)
    sample_size = min(sample_size, len(X_test))
    sample_indices = np.random.choice(X_test.index, size = sample_size, replace =False)
    X_sample = X_test.loc[sample_indices]

    # Create prediction function for calibrated model
    def predict_proba_calibrated(X):
        return calibrated_model.predict_proba(X)[:,1]
    
    # Background dataset
    background = shap.sample(X_train, n_background)

    # KernelSHAP dataset
    explainer = shap.KernelExplainer(predict_proba_calibrated, background)
    shap_values = explainer.shap_values(X_sample, nsamples = nsamples)

    # Verify efficiency axiom
    idx = 0
    pred = calibrated_model.predict_proba(X_sample.iloc[idx:idx+1])[:,1][0]
    base = explainer.expected_value
    shap_sum = shap_values[idx].sum()

    if abs(pred - (base + shap_sum)) < 0.01:
        print(" Efficiency axiom satisfied")
    else:
        print(" Note: KernelSHAP is approximate, small differences expected")
    
    return explainer, shap_values, X_sample

def calculate_feature_importance(shap_values, feature_names):
    """
    Calculate global feature importance from SHAP values

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names

    Returns
    -------
    pd.DataFrame
        Feature importance sorted by mean |SHAP|
    """
    mean_abs_shap = np.abs(shap_values).mean(axis =0)
    importance_df = pd.DataFrame({
        'Feature' : feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP',ascending=False)
    return importance_df

def explain_individual_prediction(shap_values, X_sample,
                                  y_sample, calibrated_model,
                                  idx = None):
    """
    Explain individual prediction with highest risk

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    X_sample : pd.DataFrame
        Sample data
    y_sample : np.ndarray
        Sample labels
    calibrated_model
        Calibrated model
    idx : int, optional
        Index to explain (if None, uses highest risk prediciton)

    Returns
    ----
    pd.DataFrame
        Feature contributions for the prediction
    """
    # Find high-risk prediction if idx not provided
    if idx is None:
        y_pred = calibrated_model.predict_proba(X_sample)[:,1]
        idx = y_pred.argmax()
    else:
        y_pred = calibrated_model.predict_proba(X_sample)[:,1]
    
    # SHAP values for this prediction
    shap_single = shap_values[idx]
    feature_vals = X_sample.iloc[idx]

    contribution_df = pd.DataFrame({
        'Feature' : X_sample.columns,
        'Value' : feature_vals.values,
        'SHAP_Value' : shap_single,
        'Abs_SHAP' : np.abs(shap_single)
    }).sort_values('Abs_SHAP', ascending =False)
    
    return contribution_df


# ============================================================================
# SECTION 3: ANOMALY DETECTION FUNCTIONS
# ============================================================================
def train_anomaly_detector(X_train, contamination =0.05,
                           n_estimators =100, max_samples =256,
                           random_state =42):
    """
    Train Isolation Forest for anomaly detection

    Parameters
    ----------
    X_train :pd.DataFrame
        Training data
    contamination : float, default =0.05
        Expected proportion of outlier (5%)
    n_estimators : int, default = 100
        Number of trees
    max_samples : int, default =256
        Number of samples per tree
    random _state : int, default =42
        Random seed

    Returns
    -------
    IsolationForest
        Trained anomaly detecter
    """
    iso_forest = IsolationForest(
        contamination = contamination,
        n_estimators = n_estimators,
        max_samples = max_samples,
        random_state = random_state,
        n_jobs =1
    )
    iso_forest.fit(X_train)

    return iso_forest

def detect_anomalies(iso_forest, datasets):
    """
    Detect anomalies in multiple datasets

    Parameters
    ----------
    iso_forest : IsolationForest
        Trained anomaly detector
    datasets : dict
        Dictionary of {dataset_name : DataFrame}

    Returns
    --------
    dict
        Anomaly results for each dataset
        Keys : dataset_name
        Values : dict with 'predictions', 'scores', 'rate'
    """
    anomaly_results ={}

    for name, data in datasets.items():
        predictions = iso_forest.predict(data)
        scores = iso_forest.score_samples(data)

        n_total = len(predictions)
        n_normal = (predictions ==1).sum()
        n_anomaly = (predictions == -1).sum()
        anomaly_rate = n_anomaly/ n_total
        
        anomaly_results[name] = {
            'predictions': predictions,
            'scores': scores,
            'rate': anomaly_rate
        }

    return anomaly_results
    
def analyze_anomaly_performance(anomaly_results, y_test,
                                    calibrated_model, X_test):
    """
    Analyze model performance on normal vs anomalous observations

    Parameters
    ----------
    anomaly_results : dict
        results from detect_anomalies()
    y_test : np.ndarray
        Test labels
    calibrated_model
        Calibrated model
    X_test : pd.DataFrame
        Test features

    Returns
    -------
    tuple
        (default_rate_normal, default_rate_anomaly, auc_normal, auc_anomaly, ece_normal, ece_anomaly)
    """
    # Test set analysis
    test_anomalies = anomaly_results['Test']['predictions']
    test_normal_mask = test_anomalies ==1
    test_anomaly_mask = test_anomalies == -1

    # Calibrated predictions
    y_pred_cal_test = calibrated_model.predict_proba(X_test)[:,1]
        
    # Initialize
    default_rate_normal = None
    default_rate_anomaly = None
    auc_normal = None
    auc_anomaly = None
    ece_normal = None
    ece_anomaly = None

    if test_anomaly_mask.sum() > 5: # Need enough anomalies
        # Normal observations
        y_test_normal = y_test[test_normal_mask]
        y_pred_normal = y_pred_cal_test[test_normal_mask]
        auc_normal = roc_auc_score(y_test_normal, y_pred_normal) if len(np.unique(y_test_normal)) >1 else np.nan
        ece_normal = calculate_ece(y_test_normal, y_pred_normal)
        default_rate_normal = y_test_normal.mean()

        # Anomalous observations
        y_test_anomaly = y_test[test_anomaly_mask]
        y_pred_anomaly = y_pred_cal_test[test_anomaly_mask]
        auc_anomaly = roc_auc_score(y_test_anomaly, y_pred_anomaly) if len(np.unique(y_test_anomaly)) > 1 else np.nan
        ece_anomaly = calculate_ece(y_test_anomaly, y_pred_anomaly)
        default_rate_anomaly = y_test_anomaly.mean()
    else:
        print(" Not enough anomalous observations for meaningful comparison")

    return (default_rate_normal, default_rate_anomaly, auc_normal, 
            auc_anomaly, ece_normal, ece_anomaly)
