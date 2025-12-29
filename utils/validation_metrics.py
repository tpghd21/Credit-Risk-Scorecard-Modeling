import numpy as np
import pandas as pd

def calculate_ece(y_true, y_pred, n_bins =10):
    """
    Parameters
    y_true : np.darray (0 or 1)
    y_pred : np.darray (0~1)
    n_bins: int, default =10

    Returns
    float
        ECE: Values (lower is better)
            - <0.05: Well calibrated
            - 0.05-0.15 : Fair
            - >0.15: Poor
    """
    bin_boundaries = np.linspace(0,1, n_bins+1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece =0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin >0:
            accuracy_in_bin = y_true[in_bin].mean()
            confidence_in_bin = y_pred[in_bin].mean()
            ece += prop_in_bin * abs(accuracy_in_bin - confidence_in_bin)
    
    return ece

def calculate_psi(expected, actual, bins =10, epsilon =1e-6):
    """
    Population Stability Index (PSI) computation
    
    We measure the stability between two distributions
    
    Parameters
    ----------
    expected : np.ndarray (Baseline data e.g. Train set)
    actual : np.darray(data to compare, e.g. Validation set, Test set)
    bins : int, default =10
    epsilon : float , default =1e-6 (Smoothing constant)

    Returns
    -------
    float
        PSI value
            - <0.1 : Stable
            - 0.1 - 0.25 : Monitor
            - >= 0.25 : Action required
    """
    # Percentile-based binning
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    # 구간이 너무 적으면 PSI 계산 불가
    if len(breakpoints) <= 2:
        return 0.0
    
    # 각 데이터를 구간에 할당
    expected_bins = np.digitize(expected, breakpoints[1:-1])
    actual_bins = np.digitize(actual, breakpoints[1:-1])
    
    # 각 구간의 비율 계산 (Laplace smoothing 적용)
    n_bins = len(breakpoints) - 1
    expected_percents = [(expected_bins == i).sum() + epsilon for i in range(n_bins)]
    actual_percents = [(actual_bins == i).sum() + epsilon for i in range(n_bins)]
    
    # 정규화
    expected_percents = np.array(expected_percents) / sum(expected_percents)
    actual_percents = np.array(actual_percents) / sum(actual_percents)
    
    # PSI 계산
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi

def calculate_feature_psi(X_train, X_test, bins =10):
    """ 
    Calculate PSI of all features

    Parameters
    ----------
    X_train : pd.DataFrame
        Train data
    X_test : pd.DataFrame
        Test data
    bins : int, default =10

    Returns
    -------
    pd.DataFrame
        Data frame contains PSI of each features
        Columns: ['Feature', 'PSI']
    """
    psi_features = {}
    for col in X_train.columns:
        psi = calculate_psi(X_train[col].values, X_test[col].values, bins=bins)
        psi_features[col] = psi
    
    psi_df = pd.DataFrame({
        'Feature': list(psi_features.keys()),
        'PSI': list(psi_features.values())
    }).sort_values('PSI', ascending=False)
    
    return psi_df

