import os
#to hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    matthews_corrcoef
)
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

from preprocessing import split_and_preprocess
from models.dl_models import build_dnn_model

warnings.filterwarnings("ignore")

# Metric calculations

def calcular_metriques_cliniques_completes(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    total = len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    
    marcats_revisio = fp + tp
    pct_marcats = (marcats_revisio / total) * 100.0 if total > 0 else 0.0
    
    evitats_revisio = tn + fn
    pct_evitats = (evitats_revisio / total) * 100.0 if total > 0 else 0.0
    
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
        "Sensibilitat": sens,
        "Especificitat": spec,
        "PPV": ppv,
        "NPV": npv,
        "MCC": mcc,
        "Pacients_Marcats": marcats_revisio,
        "Pct_Marcats": pct_marcats,
        "Pacients_Evitats": evitats_revisio,
        "Pct_Evitats": pct_evitats,
        "Infeccions_Perdudes": fn
    }

# search of the optimal thresholds for the three requested criteria

def trobar_llindars_optimitzats(y_true, y_prob):
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
        "Sensibilitat_0.8": best_t_sens,
        "MCC": best_t_mcc,
        "Youden": best_t_youden
    }

# model training and prediction function with class weighting for imbalanced datasets

def entrenar_i_predir(model_name, tractament, X_train_tr, y_train_tr, X_eval, random_state=42):
    n_neg = np.sum(y_train_tr == 0)
    n_pos = np.sum(y_train_tr == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    if model_name == "RegressioLogisticaPenalitzada":
        cw = 'balanced' if tractament == "Ponderacio" else None
        model = LogisticRegression(penalty='l2', C=1.0, class_weight=cw, random_state=random_state)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "QuadraticDiscriminantAnalysis":
        priors_opt = [0.5, 0.5] if tractament == "Ponderacio" else None
        model = QuadraticDiscriminantAnalysis(reg_param=0.1, priors=priors_opt)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "RandomForestClassifier":
        cw = 'balanced' if tractament == "Ponderacio" else None
        model = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight=cw, random_state=random_state)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "XGBClassifier":
        spw = scale_pos_weight if tractament == "Ponderacio" else 1.0
        model = XGBClassifier(learning_rate=0.1, max_depth=3, scale_pos_weight=spw, random_state=random_state)
        model.fit(X_train_tr, y_train_tr)
        return model.predict_proba(X_eval)[:, 1]

    elif model_name == "DenseNeuralNet":
        dnn_model = build_dnn_model(input_dim=X_train_tr.shape[1], dropout_rate=0.5, lr=1e-4)
        cw_dict = {0: 1.0, 1: float(scale_pos_weight)} if tractament == "Ponderacio" else None
        
        dnn_model.fit(
            X_train_tr, y_train_tr, 
            epochs=20, batch_size=16, 
            class_weight=cw_dict, 
            verbose=0
        )
        return dnn_model.predict(X_eval, verbose=0).flatten()

    else:
        raise ValueError(f"Model no reconegut: {model_name}")

# Boostraping loop to estimate optimism and correct performance metrics

def main(config_path, n_bootstraps=100):
    with open(config_path, "r") as f:
        config = json.load(f)

    base_dir = os.path.dirname(config_path)
    dataset_path = os.path.join(base_dir, config["dataset_path"])
    
    print("--- 1. Carregant dades originals ---")
    X_train, X_test, y_train, y_test, _, _ = split_and_preprocess(
        dataset_path,
        exclude_columns=config.get("exclude_columns", []),
        target_column=config["target_column"],
        test_size=config["test_size"],
        random_state=config["random_state"],
        normalize=config["normalize"],
        apply_smote=False
    )

    models_list = [
        "RegressioLogisticaPenalitzada",
        "QuadraticDiscriminantAnalysis",
        "RandomForestClassifier",
        "XGBClassifier",
        "DenseNeuralNet"
    ]
    
    tractaments_list = ["Sense_Tractament", "Ponderacio", "SMOTE"]
    criteris_llindar = ["Sensibilitat_0.8", "MCC", "Youden"]

    # mesuring the original performance on the training data without replacement
    print("\n--- 2. Calculant el rendiment original aparent ---")
    rendiment_original = {}

    for tractament in tractaments_list:
        if tractament == "SMOTE":
            smote = SMOTE(random_state=config["random_state"])
            X_tr_orig, y_tr_orig = smote.fit_resample(X_train, y_train)
        else:
            X_tr_orig, y_tr_orig = X_train.copy(), y_train.copy()

        for model_name in models_list:
            y_prob_orig = entrenar_i_predir(
                model_name, tractament, 
                X_tr_orig, y_tr_orig, X_train, 
                random_state=config["random_state"]
            )
            llindars_orig = trobar_llindars_optimitzats(y_train, y_prob_orig)

            for criteri in criteris_llindar:
                t_orig = llindars_orig[criteri]
                m_orig = calcular_metriques_cliniques_completes(y_train, y_prob_orig, t_orig)
                m_orig["Llindar_Aplicat"] = t_orig
                rendiment_original[(model_name, tractament, criteri)] = m_orig

    # boostrapwith the same model and treatment to estimate optimism
    print(f"\n--- 3. Executant Bootstrap ({n_bootstraps} iteracions) per calcular l'optimisme ---")
    optimisme_raw = []

    for b in range(n_bootstraps):
        if (b + 1) % 10 == 0 or b == 0:
            print(f"Ronda Bootstrap {b + 1}/{n_bootstraps}...")

        X_boot, y_boot = resample(
            X_train, y_train, 
            replace=True, 
            stratify=y_train, 
            random_state=config["random_state"] + b
        )

        for tractament in tractaments_list:
            if tractament == "SMOTE":
                try:
                    smote = SMOTE(random_state=config["random_state"] + b)
                    X_tr_boot, y_tr_boot = smote.fit_resample(X_boot, y_boot)
                except Exception:
                    X_tr_boot, y_tr_boot = X_boot.copy(), y_boot.copy()
            else:
                X_tr_boot, y_tr_boot = X_boot.copy(), y_boot.copy()

            for model_name in models_list:
                # evaluate the model on the bootstrap sample (Perform_boot)
                y_prob_on_boot = entrenar_i_predir(
                    model_name, tractament, 
                    X_tr_boot, y_tr_boot, X_boot, 
                    random_state=config["random_state"] + b
                )
                llindars_boot = trobar_llindars_optimitzats(y_boot, y_prob_on_boot)

                #evaluate the model on the original training data (Perform_orig)
                y_prob_on_orig = entrenar_i_predir(
                    model_name, tractament, 
                    X_tr_boot, y_tr_boot, X_train, 
                    random_state=config["random_state"] + b
                )

                for criteri in criteris_llindar:
                    t_b = llindars_boot[criteri]
                    

                    m_boot = calcular_metriques_cliniques_completes(y_boot, y_prob_on_boot, t_b)
                    m_orig_test = calcular_metriques_cliniques_completes(y_train, y_prob_on_orig, t_b)

                    dict_opt = {
                        "Model": model_name,
                        "Tractament": tractament,
                        "Criteri_Llindar": criteri,
                        "Llindar_Aplicat": t_b
                    }
                    for k in m_boot.keys():
                        dict_opt[k] = m_boot[k] - m_orig_test[k]

                    optimisme_raw.append(dict_opt)

    #Agregation and correction of optimism
    df_opt = pd.DataFrame(optimisme_raw)
    cols_metriques = [
        "AUROC", "AUPRC", "TP", "FP", "TN", "FN",
        "Sensibilitat", "Especificitat", "PPV", "NPV", "MCC",
        "Pacients_Marcats", "Pct_Marcats", "Pacients_Evitats", "Pct_Evitats", "Infeccions_Perdudes"
    ]

    rows_resum = []
    grouped = df_opt.groupby(["Model", "Tractament", "Criteri_Llindar"])

    for (model, tractament, criteri), group in grouped:
        orig_m = rendiment_original[(model, tractament, criteri)]
        row_dict = {
            "Model": model,
            "Tractament": tractament,
            "Criteri_Llindar": criteri,
            "Llindar_Aplicat": round(orig_m["Llindar_Aplicat"], 4)
        }

        for col in cols_metriques:
            opt_vals = group[col].values
            mean_opt = np.mean(opt_vals)
            
            val_corregit = orig_m[col] - mean_opt
            
            ic_inf = np.percentile(orig_m[col] - opt_vals, 2.5)
            ic_sup = np.percentile(orig_m[col] - opt_vals, 97.5)

            is_count = col in ["TP", "FP", "TN", "FN", "Pacients_Marcats", "Pacients_Evitats", "Infeccions_Perdudes"]
            dec = 2 if is_count else 4

            row_dict[f"{col}_Aparent"] = round(orig_m[col], dec)
            row_dict[f"{col}_Optimisme"] = round(mean_opt, dec)
            row_dict[f"{col}_Corregit"] = round(val_corregit, dec)
            row_dict[f"{col} (IC 95%)"] = f"{val_corregit:.{dec}f} ({ic_inf:.{dec}f} - {ic_sup:.{dec}f})"

        rows_resum.append(row_dict)

    df_resum = pd.DataFrame(rows_resum)

    output_csv = os.path.join(base_dir, "taula_resultats_bootstrap.csv")
    df_resum.to_csv(output_csv, index=False, sep=",")

    print("\n=======================================================")
    print(f" [OK] Càlcul d'Optimisme finalitzat amb èxit!")
    print(f" Taula corregida desada a l'arrel: {output_csv}")
    print("=======================================================\n")
    
    cols_preview = ["Model", "Tractament", "Criteri_Llindar", "Sensibilitat (IC 95%)", "Especificitat (IC 95%)", "AUROC (IC 95%)"]
    print(df_resum[cols_preview].head(10).to_string())

if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
    main(config_path, n_bootstraps=100)