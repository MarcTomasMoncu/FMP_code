import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings

# Importacions del teu propi projecte
from preprocessing import split_and_preprocess
from models.ml_models import initialize_models, train_and_evaluate_model
from models.dl_models import build_dnn_model, train_and_evaluate_dnn

warnings.filterwarnings("ignore")

def main(config_path):
    # 1. Carregar configuració
    with open(config_path, "r") as f:
        config = json.load(f)

    base_dir = os.path.dirname(config_path)
    dataset_path = os.path.join(base_dir, config["dataset_path"])
    artifacts_path = os.path.join(base_dir, config["artifacts_path"])
    results_path = os.path.join(base_dir, config["results_path"])
    os.makedirs(results_path, exist_ok=True)

    # 2. Preprocessat net (sense leakage de SMOTE a la validació)
    print("Carregant i preprocessant les dades...")
    X_train, X_test, y_train, y_test, _, _ = split_and_preprocess(
        dataset_path,
        exclude_columns=config.get("exclude_columns", []),
        target_column=config["target_column"],
        test_size=config["test_size"],
        random_state=config["random_state"],
        normalize=config["normalize"],
        apply_smote=False
    )

    # Aplicar SMOTE només al train final
    if config.get("apply_smote", True):
        smote = SMOTE(random_state=config["random_state"])
        X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
    else:
        X_train_final, y_train_final = X_train.copy(), y_train.copy()

    # 3. Entrenar models un sol cop per obtenir les probabilitats del test set
    predictions_probs = {}
    
    # Models de Machine Learning
    ml_models = initialize_models()
    for model_name, model in ml_models.items():
        print(f"Entrenant model per simulació: {model_name}...")
        tmp_file = os.path.join(artifacts_path, f"tmp_explore_{model_name}.pkl")
        _, y_test_pred_probs = train_and_evaluate_model(
            model, X_train_final.values, y_train_final, X_test.values, y_test, tmp_file
        )
        predictions_probs[model_name] = y_test_pred_probs
        if os.path.exists(tmp_file): os.remove(tmp_file)

    # Model Deep Learning (Xarxa Neuronal)
    print("Entrenant xarxa neuronal per simulació: DenseNeuralNet...")
    dnn_model = build_dnn_model(input_dim=X_train.shape[1], dropout_rate=0.5, lr=1e-4)
    tmp_file_dnn = os.path.join(artifacts_path, "tmp_explore_DNN.keras")
    _, y_test_pred_probs_dnn = train_and_evaluate_dnn(
        dnn_model, X_train_final.values, y_train_final, X_test.values, y_test, tmp_file_dnn
    )
    predictions_probs["DenseNeuralNet"] = y_test_pred_probs_dnn
    if os.path.exists(tmp_file_dnn): os.remove(tmp_file_dnn)

    # 4. Càlcul de mètriques avançant cada 0.1 de Sensibilitat (Recall)
    target_sensitivities = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    threshold_grid = np.linspace(1.0, 0.0, 201)       # Graella de precisió per al llindar
    
    plot_data = {}
    csv_rows = []

    # Totals de referència del conjunt de test per calcular els %
    total_pacients = len(y_test)
    total_infectats_reals = int(y_test.sum())

    for model_name, y_prob in predictions_probs.items():
        model_results = {
            "sens_eix_x": [], "threshold": [], "actual_recall": [], "specificity": [], 
            "pct_positives": [], "pct_false_negatives": []  # Guardarem percentatges per al gràfic
        }
        
        for target_sens in target_sensitivities:
            chosen_thresh = 0.0
            chosen_rec = 1.0
            chosen_spec = 0.0
            chosen_pos = total_pacients
            chosen_fn = 0
            
            for thresh in threshold_grid:
                y_pred = (y_prob >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                pos_predicted = int(y_pred.sum())
                
                if rec >= target_sens:
                    chosen_thresh = thresh
                    chosen_rec = rec
                    chosen_spec = spec
                    chosen_pos = pos_predicted
                    chosen_fn = int(fn)
                    break
            
            # Càlcul de percentatges clau
            pct_marcats = (chosen_pos / total_pacients * 100) if total_pacients > 0 else 0
            pct_errors = (chosen_fn / total_infectats_reals * 100) if total_infectats_reals > 0 else 0

            # Guardem les estructures per al gràfic
            model_results["sens_eix_x"].append(target_sens)
            model_results["threshold"].append(chosen_thresh)
            model_results["actual_recall"].append(chosen_rec)
            model_results["specificity"].append(chosen_spec)
            model_results["pct_positives"].append(pct_marcats)
            model_results["pct_false_negatives"].append(pct_errors)
            
            # Guardem per al fitxer CSV (amb columnes absolutes i columnes de %)
            csv_rows.append({
                "Model": model_name,
                "Sensibilitat_Objectiu": round(target_sens, 1),
                "Llindar_Tall_Threshold": round(chosen_thresh, 4),
                "Sensibilitat_Real_Obtinguda": round(chosen_rec, 4),
                "Especificitat_Obtinguda": round(chosen_spec, 4),
                "Casos_Marcats_Com_Positius": chosen_pos,
                "Pct_Marcats_Com_Positius": round(pct_marcats, 2),
                "Infeccions_No_Detectades_Errors": chosen_fn,
                "Pct_Infeccions_No_Detectades_Errors": round(pct_errors, 2)
            })
            
        plot_data[model_name] = model_results

    # 5. Exportar les mètriques i percentatges a CSV
    df_metrics = pd.DataFrame(csv_rows)
    output_csv_path = os.path.join(results_path, "exploracio_sensibilitat_metrics.csv")
    df_metrics.to_csv(output_csv_path, index=False, sep=",")
    print(f"\n[OK] Fitxer CSV desat amb percentatges a: {output_csv_path}")

    # 6. Generar el gràfic net (sense números absoluts, només % a l'eix dret)
    print("Generant gràfic d'exploració basat en Percentatges (%) ...")
    n_models = len(plot_data)
    fig, axes = plt.subplots(n_models, 1, figsize=(11, 4.5 * n_models), sharex=True)
    
    if n_models == 1:
        axes = [axes]
        
    for ax, (model_name, data) in zip(axes, plot_data.items()):
        # Eix esquerre: Ràtios (0 a 1)
        ax.plot(data["sens_eix_x"], data["threshold"], label="Llindar (Threshold)", color="blue", marker="o")
        ax.plot(data["sens_eix_x"], data["actual_recall"], label="Recall (Sensibilitat)", color="green", linestyle="--", marker="s")
        ax.plot(data["sens_eix_x"], data["specificity"], label="Especificitat", color="red", marker="^")
        
        ax.set_ylabel("Valor Ràtios (0.0 - 1.0)", color="black")
        ax.set_title(f"Exploració de Llindars i Impacte Clínic - {model_name}")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_ylim(-0.05, 1.05)
        
        # Eix dret: Únicament Percentatges (0% a 100%)
        ax2 = ax.twinx()
        ax2.plot(data["sens_eix_x"], data["pct_positives"], label="% Casos Marcats com Positius", color="purple", linestyle="-.", marker="d")
        ax2.plot(data["sens_eix_x"], data["pct_false_negatives"], label="% Infeccions NO Detectades (Errors)", color="orange", linestyle=":", marker="x", markersize=8, linewidth=2)
        
        ax2.set_ylabel("Escala de Percentatges (%)", color="black")
        ax2.tick_params(axis='y', labelcolor="black")
        ax2.set_ylim(-5, 105) # Forcem els topalls de percentatge per netedat visual
        
        # Unificar les dues llegendes en una sola caixa lateral
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="center left")

    plt.xlabel("Sensibilitat Objectiu (Eix X - Intervals 0.1)")
    plt.xticks(target_sensitivities)
    plt.tight_layout()
    
    # Desar el gràfic final
    output_plot_path = os.path.join(results_path, "exploracio_sensibilitat_llindar.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"[OK] Gràfic de percentatges completat a: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
    main(config_path)