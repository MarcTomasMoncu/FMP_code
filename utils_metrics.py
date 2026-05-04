import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def find_optimal_threshold(y_true, y_pred_probs, target_sensitivity=0.90):
    """Busca el llindar més alt que mantingui la sensibilitat objectiu."""
    best_threshold = 0.5
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (y_pred_probs >= t).astype(int)
        if recall_score(y_true, preds, zero_division=0) >= target_sensitivity:
            best_threshold = t
    return best_threshold

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred_labels = (y_pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
    
    metrics = {
        'threshold_used': threshold,
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels, zero_division=0),
        'recall': recall_score(y_true, y_pred_labels, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_probs),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0
    }
    return metrics

def cv_metrics_to_df(model_name, metrics_list, results_df):
    df = pd.DataFrame(metrics_list)
    means = df.mean()
    data = {}
    for metric in df.columns:
        data[f"{metric}_mean"] = means[metric]
    return pd.concat([results_df, pd.DataFrame(data, index=[model_name])], axis=0)