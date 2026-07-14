import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

# Importacions del teu projecte
from preprocessing import split_and_preprocess
from models.ml_models import initialize_models, train_and_evaluate_model
from models.dl_models import build_dnn_model, train_and_evaluate_dnn

warnings.filterwarnings("ignore")

def calcular_metriques_cliniques(y_true, y_prob, threshold):
    """Calcula totes les mètriques sol·licitades pel professor per a un llindar donat."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total = len(y_true)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    marcats_revisio = fp + tp
    pct_marcats = (marcats_revisio / total) * 100
    
    evitats_revisio = tn + fn
    pct_evitats = (evitats_revisio / total) * 100
    
    infeccions_perdudes = fn
    
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Sensibilitat": round(sens, 4),
        "Especificitat": round(spec, 4),
        "PPV": round(ppv, 4),
        "NPV": round(npv, 4),
        "Pacients_Marcats": marcats_revisio,
        "Pct_Marcats": round(pct_marcats, 2),
        "Pacients_Evitats": evitats_revisio,
        "Pct_Evitats": round(pct_evitats, 2),
        "Infeccions_Perdudes": infeccions_perdudes
    }

def main(config_path):
    # 1. Carregar configuració
    with open(config_path, "r") as f:
        config = json.load(f)

    base_dir = os.path.dirname(config_path)
    dataset_path = os.path.join(base_dir, config["dataset_path"])
    artifacts_path = os.path.join(base_dir, config["artifacts_path"])
    results_path = os.path.join(base_dir, config["results_path"])
    os.makedirs(results_path, exist_ok=True)

    # 2. Preprocessat Net (sense SMOTE inicial per controlar el pipeline manualment)
    print("Carregant dades originals del pipeline...")
    X_train_full, X_test, y_train_full, y_test, _, _ = split_and_preprocess(
        dataset_path,
        exclude_columns=config.get("exclude_columns", []),
        target_column=config["target_column"],
        test_size=config["test_size"],
        random_state=config["random_state"],
        normalize=config["normalize"],
        apply_smote=False  # El desactivem aquí per fer l'split de validació net
    )

    # 3. Crear conjunt de Validació (per escollir els llindars sense leakage del Test)
    X_train_fold, X_val, y_train_fold, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=config["random_state"], 
        stratify=y_train_full
    )

    # Aplicar SMOTE només a la part d'entrenament estricta
    if config.get("apply_smote", True):
        print("Aplicant SMOTE exclusivament al subset d'entrenament...")
        smote = SMOTE(random_state=config["random_state"])
        X_train_ready, y_train_ready = smote.fit_resample(X_train_fold, y_train_fold)
    else:
        X_train_ready, y_train_ready = X_train_fold.copy(), y_train_fold.copy()

    # 4. Entrenar models i obtenir probabilitats (Validació i Test)
    probs_val = {}
    probs_test = {}
    
    # Inicialitzar models de ML
    ml_models = initialize_models()
    for model_name, model in ml_models.items():
        print(f"Entrenant i generant probabilitats per a: {model_name}...")
        tmp_file = os.path.join(artifacts_path, f"tmp_eval_{model_name}.pkl")
        
        # Entrenem amb el train net amb SMOTE, obtenim probabilitats de Val i de Test
        # Ens assegurem de demanar la probabilitat de la classe positiva [:, 1] internament o forçant-ho aquí
        _, p_test = train_and_evaluate_model(model, X_train_ready.values, y_train_ready, X_test.values, y_test, tmp_file)
        
        # Per a la validació, tornem a predir sobre el model entrenat
        if hasattr(model, "predict_proba"):
            p_val = model.predict_proba(X_val.values)[:, 1]
        else:
            p_val = model.predict(X_val.values) # Casos rars sense predict_proba
            
        probs_val[model_name] = p_val
        probs_test[model_name] = p_test
        if os.path.exists(tmp_file): os.remove(tmp_file)

    # Model Deep Learning (DNN)
    print("Entrenant xarxa neuronal: DenseNeuralNet...")
    dnn_model = build_dnn_model(input_dim=X_train_ready.shape[1], dropout_rate=0.5, lr=1e-4)
    tmp_file_dnn = os.path.join(artifacts_path, "tmp_eval_DNN.keras")
    _, p_test_dnn = train_and_evaluate_dnn(dnn_model, X_train_ready.values, y_train_ready, X_test.values, y_test, tmp_file_dnn)
    p_val_dnn = dnn_model.predict(X_val.values).flatten()
    
    probs_val["DenseNeuralNet"] = p_val_dnn
    probs_test["DenseNeuralNet"] = p_test_dnn
    if os.path.exists(tmp_file_dnn): os.remove(tmp_file_dnn)

    # 5. Càlcul de Llindars en VALIDACIÓ i Avaluació en TEST
    informe_final_rows = []
    threshold_grid = np.linspace(0.0, 1.0, 1001)

    for model_name in probs_val.keys():
        p_v = probs_val[model_name]
        p_t = probs_test[model_name]
        
        # Calcular mètriques globals (AUROC i AUPRC) directament en Test
        auroc = roc_auc_score(y_test, p_t)
        precision, recall, _ = precision_recall_curve(y_test, p_t)
        auprc = auc(recall, precision)
        
        # --- CRITERIS DE SELECCIÓ DE LLINDAR (Sobre Validació) ---
        
        # Criteri 1: Estàndard 0.5
        t_std = 0.5
        
        # Criteri 2: Alta Sensibilitat >= 90% (Maximitzant Especificitat en Val)
        t_90 = 0.5
        best_spec_90 = -1
        for t in threshold_grid:
            y_pred_v = (p_v >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_v).ravel()
            sens_v = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec_v = tn / (tn + fp) if (tn + fp) > 0 else 0
            if sens_v >= 0.90 and spec_v > best_spec_90:
                best_spec_90 = spec_v
                t_90 = t
                
        # Criteri 2b: Alta Sensibilitat >= 95%
        t_95 = 0.5
        best_spec_95 = -1
        for t in threshold_grid:
            y_pred_v = (p_v >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_v).ravel()
            sens_v = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec_v = tn / (tn + fp) if (tn + fp) > 0 else 0
            if sens_v >= 0.95 and spec_v > best_spec_95:
                best_spec_95 = spec_v
                t_95 = t

        # Criteri 3: Optimitzat per utilitat clínica (Youden com a referència balancejada)
        t_youden = 0.5
        best_youden = -2
        for t in threshold_grid:
            y_pred_v = (p_v >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_v).ravel()
            sens_v = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec_v = tn / (tn + fp) if (tn + fp) > 0 else 0
            current_youden = sens_v + spec_v - 1
            if current_youden > best_youden:
                best_youden = current_youden
                t_youden = t

        # Diccionari amb els llindars fixats a Validació per avaluar-los a Test
        llindars_a_testari = {
            "Standard (0.5)": t_std,
            "Alta Sensibilitat (>=90%)": t_90,
            "Alta Sensibilitat (>=95%)": t_95,
            "Optimitzat Clinic (Youden)": t_youden
        }
        
        # També afegim els rangs fixos que demana de 80% i 85% a validació
        for sens_target in [0.80, 0.85]:
            t_target = 0.5
            best_spec_target = -1
            for t in threshold_grid:
                y_pred_v = (p_v >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred_v).ravel()
                sens_v = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec_v = tn / (tn + fp) if (tn + fp) > 0 else 0
                if sens_v >= sens_target and spec_v > best_spec_target:
                    best_spec_target = spec_v
                    t_target = t
            llindars_a_testari[f"Rang Sensibilitat (>={int(sens_target*100)}%)"] = t_target

        # --- AVALUACIÓ REAL SOBRE EL TEST SET ---
        for criteri, threshold_escollit in llindars_a_testari.items():
            metrics = calcular_metriques_cliniques(y_test, p_t, threshold_escollit)
            metrics.update({
                "Model": model_name,
                "Criteri_Llindar": criteri,
                "Llindar_Val": round(threshold_escollit, 4),
                "AUROC_Test": round(auroc, 4),
                "AUPRC_Test": round(auprc, 4)
            })
            informe_final_rows.append(metrics)

    # 6. Guardar la Taula de Resultats Sol·licitada
    df_informe = pd.DataFrame(informe_final_rows)
    # Reordenem columnes per a una lectura impecable
    column_order = [
        "Model", "Criteri_Llindar", "Llindar_Val", "AUROC_Test", "AUPRC_Test",
        "TP", "FP", "TN", "FN", "Sensibilitat", "Especificitat", "PPV", "NPV",
        "Pacients_Marcats", "Pct_Marcats", "Pacients_Evitats", "Pct_Evitats", "Infeccions_Perdudes"
    ]
    df_informe = df_informe[column_order]
    
    csv_output = os.path.join(results_path, "avaluacio_clinica_final.csv")
    df_informe.to_csv(csv_output, index=False)
    print(f"\n[OK] Taula final guardada correctament a: {csv_output}")
    print(df_informe[[ "Model", "Criteri_Llindar", "Llindar_Val", "Sensibilitat", "Especificitat", "Pct_Evitats", "Infeccions_Perdudes" ]].to_string())

    # 7. GENERAR GRÀFIC DE DISTRIBUCIÓ DE PROBABILITATS (Infectats vs No Infectats)
    print("\nGenerant gràfics de distribució de probabilitats predites...")
    n_models = len(probs_test)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1: axes = [axes]
    
    for ax, (model_name, p_t) in zip(axes, probs_test.items()):
        # Separar probabilitats segons la realitat del pacient
        prob_infectats = p_t[y_test == 1]
        prob_sants = p_t[y_test == 0]
        
        ax.hist(prob_sants, bins=30, alpha=0.6, color="blue", label="No Infectats (0)", density=True)
        ax.hist(prob_infectats, bins=30, alpha=0.6, color="red", label="Infectats (1)", density=True)
        
        ax.set_title(f"Distribución: {model_name}")
        ax.set_xlabel("Probabilitat Predita (Classe 1)")
        ax.set_ylabel("Densitat")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    plot_output = os.path.join(results_path, "distribucio_probabilitats.png")
    plt.savefig(plot_output, dpi=300)
    print(f"[OK] Gràfic de distribucions desat a: {plot_output}")
    plt.show()

if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
    main(config_path)