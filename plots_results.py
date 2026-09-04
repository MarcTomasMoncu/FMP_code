import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extreure_ic(val_str):
    """
    Extreu el valor corregit i els límits de l'IC 95% del format "0.8065 (0.7188 - 0.8792)".
    """
    try:
        match = re.search(r"([-\d\.]+)\s*\(([-\d\.]+)\s*-\s*([-\d\.]+)\)", str(val_str))
        if match:
            val = float(match.group(1))
            inf = float(match.group(2))
            sup = float(match.group(3))
            return val, val - inf, sup - val
    except Exception:
        pass
    return None, 0, 0

# 1. GRÀFIC DE BARRES AGRUPADES (AUROC)
def generar_grafic_auroc_grouped(csv_path="bootstrap_results_table.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El fitxer '{csv_path}' no existeix. Comprova la ruta.")

    df = pd.read_csv(csv_path)
    df_youden = df[df["Threshold_Criterion"] == "Youden"].copy()
    if df_youden.empty:
        df_youden = df.drop_duplicates(subset=["Model", "Treatment"]).copy()

    model_map = {
        "PenalizedLogisticRegression": "Penalized Logistic Reg.",
        "QDA": "QDA",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "DNN": "DNN"
    }

    treatment_map = {
        "No_Treatment": "No Treatment",
        "Class_Weighting": "Class Weighting",
        "SMOTENC": "SMOTENC"
    }

    df_youden["Model_EN"] = df_youden["Model"].map(lambda x: model_map.get(x, x))
    df_youden["Treatment_EN"] = df_youden["Treatment"].map(lambda x: treatment_map.get(x, x))

    parsed_data = []
    for idx, row in df_youden.iterrows():
        val, err_inf, err_sup = extreure_ic(row["AUROC (95% CI)"])
        if val is None:
            val = row["AUROC_Corrected"]
            err_inf, err_sup = 0, 0
            
        parsed_data.append({
            "Model": row["Model_EN"],
            "Treatment": row["Treatment_EN"],
            "AUROC": val,
            "Err_Inf": err_inf,
            "Err_Sup": err_sup
        })

    df_plot = pd.DataFrame(parsed_data)

    models_order = [
        "Penalized Logistic Reg.",
        "QDA",
        "Random Forest",
        "XGBoost",
        "DNN"
    ]
    treatments = ["No Treatment", "Class Weighting", "SMOTENC"]

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = {
        "No Treatment": "#e74c3c",       
        "Class Weighting": "#2ecc71",    
        "SMOTENC": "#3498db"             
    }

    x = np.arange(len(models_order))
    width = 0.25

    for i, tr in enumerate(treatments):
        df_tr = df_plot[df_plot["Treatment"] == tr]
        vals, errs_inf, errs_sup = [], [], []
        for m in models_order:
            sub = df_tr[df_tr["Model"] == m]
            if not sub.empty:
                vals.append(sub["AUROC"].values[0])
                errs_inf.append(sub["Err_Inf"].values[0])
                errs_sup.append(sub["Err_Sup"].values[0])
            else:
                vals.append(0)
                errs_inf.append(0)
                errs_sup.append(0)

        offset = (i - 1) * width
        rects = ax.bar(
            x + offset, vals, width, 
            yerr=[errs_inf, errs_sup],
            capsize=4, label=tr, 
            color=palette.get(tr, "#95a5a6"),
            edgecolor="black", linewidth=0.8, alpha=0.85
        )

        for rect, val in zip(rects, vals):
            if val > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    val + 0.025, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold"
                )

    ax.set_ylabel("Optimism-Corrected AUROC (95% CI)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=10, fontweight="bold")
    ax.set_ylim([0.4, 1.02])
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random Chance (0.50)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(title="Data Treatment", title_fontsize='10', loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, "auroc_grouped_by_treatment.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Gràfic de barres d'AUROC desat a: {os.path.abspath(out_png)}")


# 2. DOBLE HEATMAP (SENSITIVITY & SPECIFICITY SIDE-BY-SIDE)
def generar_heatmap_partit(csv_path="bootstrap_results_table.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El fitxer '{csv_path}' no existeix. Comprova la ruta.")

    df = pd.read_csv(csv_path)

    model_map = {
        "PenalizedLogisticRegression": "Penalized Logistic Reg.",
        "QDA": "QDA",
        "RandomForest": "Random Forest",
        "XGBoost": "XGBoost",
        "DNN": "DNN"
    }

    treatment_map = {
        "No_Treatment": "No Treatment",
        "Class_Weighting": "Class Weighting",
        "SMOTENC": "SMOTENC"
    }

    criterion_map = {
        "MCC": "Max MCC",
        "Sensitivity_0.8": "Sensitivity ≥ 0.80",
        "Youden": "Youden Index"
    }

    df["Model_EN"] = df["Model"].map(lambda x: model_map.get(x, x))
    df["Treatment_EN"] = df["Treatment"].map(lambda x: treatment_map.get(x, x))
    df["Criterion_EN"] = df["Threshold_Criterion"].map(lambda x: criterion_map.get(x, x))

    sens_vals, esp_vals = [], []
    for idx, row in df.iterrows():
        s_val, _, _ = extreure_ic(row["Sensitivity (95% CI)"])
        if s_val is None:
            s_val = row["Sensitivity_Corrected"]
        sens_vals.append(s_val)

        e_val, _, _ = extreure_ic(row["Specificity (95% CI)"])
        if e_val is None:
            e_val = row["Specificity_Corrected"]
        esp_vals.append(e_val)

    df["Sens_Val"] = sens_vals
    df["Esp_Val"] = esp_vals
    df["Model_Treatment"] = df["Model_EN"] + " (" + df["Treatment_EN"] + ")"

    models_order = ["Penalized Logistic Reg.", "QDA", "Random Forest", "XGBoost", "DNN"]
    treatments_order = ["No Treatment", "Class Weighting", "SMOTENC"]
    
    desired_index = []
    for m in models_order:
        for t in treatments_order:
            desired_index.append(f"{m} ({t})")

    criteria_order = ["Max MCC", "Sensitivity ≥ 0.80", "Youden Index"]

    pivot_sens = df.pivot(index="Model_Treatment", columns="Criterion_EN", values="Sens_Val").reindex(desired_index)[criteria_order]
    pivot_esp = df.pivot(index="Model_Treatment", columns="Criterion_EN", values="Esp_Val").reindex(desired_index)[criteria_order]

    # Configuració del doble Heatmap
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 11), sharey=True)

    # 1. Heatmap de Sensibility
    sns.heatmap(
        pivot_sens, 
        ax=axes[0], 
        annot=True, 
        fmt=".3f", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Optimism-Corrected Sensitivity'},
        linewidths=1.2, 
        linecolor='white',
        vmin=0.0, 
        vmax=1.0
    )
    axes[0].set_title("Sensitivity by Threshold Criterion", fontsize=12, fontweight='bold', pad=15)
    axes[0].set_xlabel("Threshold Selection Criterion", fontsize=10, fontweight='bold', labelpad=10)
    axes[0].set_ylabel("Model & Data Treatment", fontsize=10, fontweight='bold', labelpad=10)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right", fontweight='bold')
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontweight='bold')

    # 2. Heatmap especificity
    sns.heatmap(
        pivot_esp, 
        ax=axes[1], 
        annot=True, 
        fmt=".3f", 
        cmap="YlOrRd", 
        cbar_kws={'label': 'Optimism-Corrected Specificity'},
        linewidths=1.2, 
        linecolor='white',
        vmin=0.0, 
        vmax=1.0
    )
    axes[1].set_title("Specificity by Threshold Criterion", fontsize=12, fontweight='bold', pad=15)
    axes[1].set_xlabel("Threshold Selection Criterion", fontsize=10, fontweight='bold', labelpad=10)
    axes[1].set_ylabel("", fontweight='bold') # Buit per estar compartit
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right", fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, "sensitivity_specificity_side_by_side_heatmap.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Doble Heatmap desat correctament a: {os.path.abspath(out_png)}")

if __name__ == "__main__":
    csv_input = "bootstrap_results_table.csv"
    generar_grafic_auroc_grouped(csv_input, output_dir="results")
    generar_heatmap_partit(csv_input, output_dir="results")