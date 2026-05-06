import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

def find_optimal_threshold(y_true, y_pred_probs, target_sensitivity=0.90): #function to find the optimal threshold for the model to achieve a target sensitivity (recall) of 90%
    best_threshold = 0.5 #default threshold
    for t in np.arange(0.01, 1.0, 0.01): #iterate over possible thresholds from 0.01 to 0.99 with a step of 0.01
        preds = (y_pred_probs >= t).astype(int) 
        if recall_score(y_true, preds, zero_division=0) >= target_sensitivity: #if the recall (sensitivity) is greater than or equal to the target sensitivity, update the best threshold
            best_threshold = t
    return best_threshold

def calculate_metrics(y_true, y_pred_probs, threshold=0.5): #function to calculate the performance metrics of the model given the true labels, predicted probabilities and a threshold to convert probabilities into binary predictions
    y_pred_labels = (y_pred_probs >= threshold).astype(int) 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel() #calculate the confusion matrix and extract true negatives, false positives, false negatives and true positives
    
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

def cv_metrics_to_df(model_name, metrics_list, results_df): #function to convert the list of metrics obtained from cross-validation into a DataFrame and calculate the mean of each metric across the folds, then concatenate it to the results_df DataFrame with the model name as index
    df = pd.DataFrame(metrics_list)
    means = df.mean()
    data = {}
    for metric in df.columns:
        data[f"{metric}_mean"] = means[metric]
    return pd.concat([results_df, pd.DataFrame(data, index=[model_name])], axis=0)