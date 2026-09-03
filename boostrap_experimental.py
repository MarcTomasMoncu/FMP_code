import os
# To hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    matthews_corrcoef,
    roc_curve
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Canvi a la nova funció que no divideix el dataset
from preprocessing import preprocess_full_dataset
from models.dl_models import build_dnn_model

warnings.filterwarnings("ignore")

# ==========================================
# 1. METRICS CALCULATION
# ==========================================
def calculate_complete_clinical_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    total = len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    
    flagged_for_review = fp + tp
    pct_flagged = (flagged_for_review / total) * 100.0 if total > 0 else 0.0
    
    avoided_review = tn + fn
    pct_avoided = (avoided_review / total) * 100.0 if total > 0 else 0.0
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)
    except Exception:
        auroc, auprc = 0.0, 0.0

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "MCC": mcc,
        "Patients_Flagged": flagged_for_review,
        "Pct_Flagged": pct_flagged,
        "Patients_Avoided": avoided_review,
        "Pct_Avoided": pct_avoided,
        "Missed_Infections": fn
    }

# ==========================================
# 2. OPTIMAL THRESHOLDS SEARCH
# ==========================================
def find_optimized_thresholds(y_true, y_prob):
    threshold_grid = np.linspace(0.0, 1.0, 501)
    
    best_t_sens, best_spec_for_sens = 0.5, -1.0
    best_t_mcc, best_mcc_val = 0.5, -2.0
    best_t_youden, best_youden_val = 0.5, -2.0
    
    for t in threshold_grid:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = matthews_corrcoef(y_true, y_pred)
        youden = sens + spec - 1.0
        
        if sens >= 0.80 and spec > best_spec_for_sens:
            best_spec_for_sens = spec
            best_t_sens = t
            
        if mcc > best_mcc_val:
            best_mcc_val = mcc
            best_t_mcc = t
            
        if youden > best_youden_val:
            best_youden_val = youden
            best_t_youden = t
            
    return {
        "Sensitivity_0.8": best_t_sens,
        "MCC": best_t_mcc,
        "Youden": best_t_youden
    }

# ==========================================
# 3. MODEL TRAINING AND PREDICTION
# ==========================================
def train_and_predict(model_name, treatment, X_train_tr, y_train_tr, X_eval, random_state=42):
    n_neg = np.sum(y_train_tr == 0)
    n_pos = np.sum(y_train_tr == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    if model_name == "PenalizedLogisticRegression":
        cw = 'balanced' if treatment == "Class_Weighting" else None
        model = LogisticRegression(penalty='l2', C=1.0, class_weight=cw, random_state=random_state)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "QDA":
        priors_opt = [0.5, 0.5] if treatment == "Class_Weighting" else None
        model = QuadraticDiscriminantAnalysis(reg_param=0.1, priors=priors_opt)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "RandomForest":
        cw = 'balanced' if treatment == "Class_Weighting" else None
        model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight=cw, random_state=random_state)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "XGBoost":
        spw = scale_pos_weight if treatment == "Class_Weighting" else 1.0
        model = XGBClassifier(learning_rate=0.1, max_depth=3, scale_pos_weight=spw, random_state=random_state)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "DNN":
        dnn_model = build_dnn_model(input_dim=X_train_tr.shape[1], dropout_rate=0.5, lr=1e-4)
        cw_dict = {0: 1.0, 1: float(scale_pos_weight)} if treatment == "Class_Weighting" else None
        
        dnn_model.fit(
            X_train_tr, y_train_tr, 
            epochs=20, batch_size=16, 
            class_weight=cw_dict, 
            verbose=0
        )
        return dnn_model.predict(X_eval, verbose=0).flatten()

    else:
        raise ValueError(f"Model not recognized: {model_name}")

# ==========================================
# 4. BOOTSTRAP MAIN PIPELINE
# ==========================================
def main(config_path, n_bootstraps=100):
    with open(config_path, "r") as f:
        config = json.load(f)

    base_dir = os.path.dirname(config_path)
    dataset_path = os.path.join(base_dir, config["dataset_path"])
    
    print("--- 1. Loading original dataset (Complete Cohort) ---")
    X_full, y_full, _, _ = preprocess_full_dataset(
        dataset_path,
        exclude_columns=config.get("exclude_columns", []),
        target_column=config["target_column"],
        random_state=config["random_state"],
        normalize=config["normalize"],
        apply_smote=False
    )

    models_list = ["PenalizedLogisticRegression", "QDA", "RandomForest", "XGBoost", "DNN"]
    treatments_list = ["No_Treatment", "Class_Weighting", "SMOTENC"]
    threshold_criteria = ["Sensitivity_0.8", "MCC", "Youden"]

    # Configuration for ROC Curves
    configs_to_plot = [
        ("PenalizedLogisticRegression", "No_Treatment"),
        ("QDA", "Class_Weighting"),
        ("RandomForest", "No_Treatment"),
        ("XGBoost", "Class_Weighting"),
        ("DNN", "SMOTENC")
    ]
    
    # Dictionary for pretty-printing in the plot
    friendly_names = {
        "PenalizedLogisticRegression": "Penalized Logistic Reg.",
        "QDA": "QDA",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "DNN": "DNN",
        "No_Treatment": "No Treatment",
        "Class_Weighting": "Class Weighting",
        "SMOTENC": "SMOTENC"
    }

    roc_data = {cfg: [] for cfg in configs_to_plot}
    mean_fpr = np.linspace(0, 1, 100)

    print("\n--- 2. Calculating apparent original performance (Complete Cohort) ---")
    original_performance = {}

    for treatment in treatments_list:
        if treatment == "SMOTENC":
            smote = SMOTE(random_state=config["random_state"])
            X_tr_orig, y_tr_orig = smote.fit_resample(X_full, y_full)
        else:
            X_tr_orig, y_tr_orig = X_full.copy(), y_full.copy()

        for model_name in models_list:
            y_prob_orig = train_and_predict(
                model_name, treatment, 
                X_tr_orig, y_tr_orig, X_full, 
                random_state=config["random_state"]
            )
            orig_thresholds = find_optimized_thresholds(y_full, y_prob_orig)

            for criterion in threshold_criteria:
                t_orig = orig_thresholds[criterion]
                m_orig = calculate_complete_clinical_metrics(y_full, y_prob_orig, t_orig)
                m_orig["Applied_Threshold"] = t_orig
                original_performance[(model_name, treatment, criterion)] = m_orig

    print(f"\n--- 3. Running Bootstrap ({n_bootstraps} iterations) for optimism correction and ROCs ---")
    raw_optimism = []

    for b in range(n_bootstraps):
        if (b + 1) % 10 == 0 or b == 0:
            print(f"Bootstrap Round {b + 1}/{n_bootstraps}...")

        X_boot, y_boot = resample(
            X_full, y_full, 
            replace=True, 
            stratify=y_full, 
            random_state=config["random_state"] + b
        )

        for treatment in treatments_list:
            if treatment == "SMOTENC":
                try:
                    smote = SMOTE(random_state=config["random_state"] + b)
                    X_tr_boot, y_tr_boot = smote.fit_resample(X_boot, y_boot)
                except Exception:
                    X_tr_boot, y_tr_boot = X_boot.copy(), y_boot.copy()
            else:
                X_tr_boot, y_tr_boot = X_boot.copy(), y_boot.copy()

            for model_name in models_list:
                # 1. Model evaluated on bootstrap sample
                y_prob_on_boot = train_and_predict(
                    model_name, treatment, 
                    X_tr_boot, y_tr_boot, X_boot, 
                    random_state=config["random_state"] + b
                )
                boot_thresholds = find_optimized_thresholds(y_boot, y_prob_on_boot)

                # 2. Model evaluated on original sample (Complete Cohort)
                y_prob_on_orig = train_and_predict(
                    model_name, treatment, 
                    X_tr_boot, y_tr_boot, X_full, 
                    random_state=config["random_state"] + b
                )
                
                # Extract ROC for selected configs
                if (model_name, treatment) in configs_to_plot:
                    fpr, tpr, _ = roc_curve(y_full, y_prob_on_orig)
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0 
                    roc_data[(model_name, treatment)].append(interp_tpr)

                # Collect metrics for optimism
                for criterion in threshold_criteria:
                    t_b = boot_thresholds[criterion]
                    
                    m_boot = calculate_complete_clinical_metrics(y_boot, y_prob_on_boot, t_b)
                    m_orig_test = calculate_complete_clinical_metrics(y_full, y_prob_on_orig, t_b)

                    dict_opt = {
                        "Model": model_name,
                        "Treatment": treatment,
                        "Threshold_Criterion": criterion,
                        "Applied_Threshold": t_b
                    }
                    for k in m_boot.keys():
                        dict_opt[k] = m_boot[k] - m_orig_test[k]

                    raw_optimism.append(dict_opt)

    # ==========================================
    # PLOTTING THE ROC CURVES
    # ==========================================
    print("\nGenerating ROC curves plot...")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, cfg in enumerate(configs_to_plot):
        ax = axes[idx]
        tprs = roc_data[cfg]
        
        # Draw bootstrap curves (light gray)
        for tpr in tprs:
            ax.plot(mean_fpr, tpr, color='gray', alpha=0.15, lw=1)
            
        # Draw mean curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0 
        mean_auc = auc(mean_fpr, mean_tpr)
        
        ax.plot(mean_fpr, mean_tpr, color='#1f77b4', lw=2.5, label=f'Mean (AUC = {mean_auc:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='red', label='Random Chance')
        
        # Plot formatting
        formatted_model = friendly_names[cfg[0]]
        formatted_treatment = friendly_names[cfg[1]]
        
        ax.set_title(f"{formatted_model}\n({formatted_treatment})", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9, fontweight='bold')
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    # Hide the empty last subplot (5 configs for 6 slots)
    axes[-1].axis('off')
    
    plt.tight_layout()
    plot_output_path = os.path.join(base_dir, "roc_curves_bootstrap.png")
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Plot saved successfully at: {plot_output_path}")

    # ==========================================
    # AGGREGATION & FINAL EXPORT
    # ==========================================
    df_opt = pd.DataFrame(raw_optimism)
    metrics_cols = [
        "AUROC", "AUPRC", "TP", "FP", "TN", "FN",
        "Sensitivity", "Specificity", "PPV", "NPV", "MCC",
        "Patients_Flagged", "Pct_Flagged", "Patients_Avoided", "Pct_Avoided", "Missed_Infections"
    ]

    rows_summary = []
    grouped = df_opt.groupby(["Model", "Treatment", "Threshold_Criterion"])

    for (model, treatment, criterion), group in grouped:
        orig_m = original_performance[(model, treatment, criterion)]
        row_dict = {
            "Model": model,
            "Treatment": treatment,
            "Threshold_Criterion": criterion,
            "Applied_Threshold": round(orig_m["Applied_Threshold"], 4)
        }

        for col in metrics_cols:
            opt_vals = group[col].values
            mean_opt = np.mean(opt_vals)
            
            corrected_val = orig_m[col] - mean_opt
            
            # 95% Confidence Intervals
            ci_inf = np.percentile(orig_m[col] - opt_vals, 2.5)
            ci_sup = np.percentile(orig_m[col] - opt_vals, 97.5)

            is_count = col in ["TP", "FP", "TN", "FN", "Patients_Flagged", "Patients_Avoided", "Missed_Infections"]
            dec = 2 if is_count else 4

            row_dict[f"{col}_Apparent"] = round(orig_m[col], dec)
            row_dict[f"{col}_Optimism"] = round(mean_opt, dec)
            row_dict[f"{col}_Corrected"] = round(corrected_val, dec)
            row_dict[f"{col} (95% CI)"] = f"{corrected_val:.{dec}f} ({ci_inf:.{dec}f} - {ci_sup:.{dec}f})"

        rows_summary.append(row_dict)

    df_summary = pd.DataFrame(rows_summary)

    output_csv = os.path.join(base_dir, "bootstrap_results_table.csv")
    df_summary.to_csv(output_csv, index=False, sep=",")

    print("\n=======================================================")
    print(f" [OK] Optimism calculation and correction completed!")
    print(f" Corrected table saved at: {output_csv}")
    print("=======================================================\n")
    
    preview_cols = ["Model", "Treatment", "Threshold_Criterion", "Sensitivity (95% CI)", "Specificity (95% CI)", "AUROC (95% CI)"]
    print(df_summary[preview_cols].head(10).to_string())

if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
    main(config_path, n_bootstraps=100)