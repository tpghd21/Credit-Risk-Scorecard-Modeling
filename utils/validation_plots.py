"""
Visualization Utilities
========================
Functions for creating validation plots and charts
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay

def plot_calibration_curves(models, calibrated_models,
                            X_test, y_test,
                            figsize=(16,12)):
    model_names = list(models.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(2, n_models, figsize = figsize)

    if n_models ==1:
        axes = axes.reshape(2,1)
    for i , name in enumerate(model_names):
        # Before calibration
        CalibrationDisplay.from_estimator(
            models[name], X_test, y_test,
            n_bins =10, name ='Before Calibration', ax = axes[0,i]
        )
        axes[0, i].set_title(f'Calibration: {name} (Before)', 
                            fontsize=14, fontweight='bold')
        axes[0, i].grid(True, alpha=0.3)
        
        # After calibration
        CalibrationDisplay.from_estimator(
            calibrated_models[name], X_test, y_test,
            n_bins=10, name='After Sigmoid Calibration', ax=axes[1, i]
        )
        axes[1, i].set_title(f'Calibration: {name} (After)', 
                            fontsize=14, fontweight='bold')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def plot_roc_and_distributions(calibrated_models, X_test,
                               y_test, figsize =(16,6)):
    fig, axes = plt.subplots(1,2, figsize = figsize)

    # ROC Curves
    for name, model in calibrated_models.items():
        RocCurveDisplay.from_estimator(
            model, X_test, y_test,
            ax = axes[0], name = f'{name} (Calibrated)'
        )
    axes[0].plot([0,1],[0,1], 'k--', alpha =0.5)
    axes[0].set_title('ROC Curves (Calibrated Models)',
                      fontsize =14, fontweight ='bold')
    axes[0].grid(True, alpha =0.3)

    # Probability distribution
    for name, model in calibrated_models.items():
        prob = model.predict_proba(X_test)[:,1]
        axes[1].hist(prob, bins =30, alpha =0.5, label =f'{name}', density = True)
    
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Predicted Probability Distributions', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_psi_analysis(train_pred, test_pred,
                      psi_df, psi_value,
                      figsize =(16,6)):
    fig, axes = plt.subplots(1,2, figsize = figsize)
    # Score distribution comparison
    axes[0].hist(train_pred, bins =30, alpha =0.5, label ='Train',
                 density= True, color ='blue')
    axes[0].hist(test_pred, bins=30, alpha=0.5, label='Test', 
                density=True, color='red')
    axes[0].set_xlabel('Calibrated Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'Score Distribution: PSI = {psi_value:.4f}', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    # Feature PSI bar chart
    colors = ['green' if x < 0.10 else 'orange' if x < 0.25 else 'red' 
              for x in psi_df['PSI']]
    axes[1].barh(psi_df['Feature'], psi_df['PSI'], color=colors, alpha=0.7)
    axes[1].set_xlabel('PSI Value', fontsize=12)
    axes[1].set_ylabel('Features', fontsize=12)
    axes[1].set_title('Population Stability Index by Feature', 
                     fontsize=14, fontweight='bold')
    axes[1].axvline(x=0.10, color='orange', linestyle='--', 
                   linewidth=2, label='Monitor (0.10)')
    axes[1].axvline(x=0.25, color='red', linestyle='--', 
                   linewidth=2, label='Action (0.25)')
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_shap_summary(shap_values, X_sample,
                     figsize=(12, 8)):
    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title('SHAP Summary - Calibrated Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_shap_feature_importance(importance_df, top_n=10,
                                 figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.barh(importance_df['Feature'].head(top_n), 
            importance_df['Mean_Abs_SHAP'].head(top_n), 
            color='steelblue', alpha=0.7)
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Features - Calibrated Model', 
             fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_shap_waterfall(shap_values, X_sample,
                       explainer, idx=0):
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[idx].values,
        feature_names=X_sample.columns.tolist()
    ))

def plot_anomaly_analysis(anomaly_results, y_test,
                         default_rate_normal=None,
                         default_rate_anomaly=None,
                         figsize=(16, 6)):
    """
    Plot anomaly detection analysis
    
    Parameters
    ----------
    anomaly_results : dict
        Results from detect_anomalies()
    y_test : np.ndarray
        Test labels
    default_rate_normal : float, optional
        Default rate for normal observations
    default_rate_anomaly : float, optional
        Default rate for anomalous observations
    figsize : tuple, default=(16, 6)
        Figure size
        
    Examples
    --------
    >>> plot_anomaly_analysis(anomaly_results, y_test, 0.28, 0.83)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    test_predictions = anomaly_results['Test']['predictions']
    test_scores = anomaly_results['Test']['scores']
    test_normal_mask = test_predictions == 1
    test_anomaly_mask = test_predictions == -1
    
    # Score distribution
    axes[0].hist(test_scores[test_normal_mask], bins=30, alpha=0.6, 
                label='Normal', color='green', density=True)
    axes[0].hist(test_scores[test_anomaly_mask], bins=30, alpha=0.6, 
                label='Anomaly', color='red', density=True)
    axes[0].set_xlabel('Anomaly Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Isolation Forest Score Distribution', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Default rate comparison
    if default_rate_normal is not None and default_rate_anomaly is not None:
        categories = ['Normal', 'Anomalous', 'All']
        rates = [default_rate_normal, default_rate_anomaly, y_test.mean()]
        colors_bar = ['green', 'red', 'blue']
        
        bars = axes[1].bar(categories, rates, color=colors_bar, alpha=0.7)
        axes[1].set_ylabel('Default Rate', fontsize=12)
        axes[1].set_title('Default Rate by Anomaly Status', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, max(rates) * 1.2])
        
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{rate:.1%}', ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient anomalies\nfor comparison',
                    ha='center', va='center', fontsize=14, 
                    transform=axes[1].transAxes)
        axes[1].set_title('Default Rate by Anomaly Status', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()