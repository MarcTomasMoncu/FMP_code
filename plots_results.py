import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

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

# ==========================================
# 1. GRÀFIC DE BARRES AGRUPADES (AUROC)
# ==========================================
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


# ==========================================
# 2. HEATMAP AMB CEL·LES PARTIDES (DIAGONAL SPLIT)
# ==========================================
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
        "Sensitivity_0.8": "Sensitivity ≥ 0.80",
        "MCC": "Max MCC",
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

    criteria_order = ["Sensitivity ≥ 0.80", "Max MCC", "Youden Index"]

    pivot_sens = df.pivot(index="Model_Treatment", columns="Criterion_EN", values="Sens_Val").reindex(desired_index)[criteria_order]
    pivot_esp = df.pivot(index="Model_Treatment", columns="Criterion_EN", values="Esp_Val").reindex(desired_index)[criteria_order]

    fig, ax = plt.subplots(figsize=(10, 13))

    nrows, ncols = pivot_sens.shape
    
    cmap_sens = plt.get_cmap("YlGnBu")
    cmap_esp = plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Dibuixar cada cel·la dividida en diagonal
    for i in range(nrows):
        for j in range(ncols):
            s_val = pivot_sens.iloc[i, j]
            e_val = pivot_esp.iloc[i, j]

            # Coordenades del quadrat
            rect = patches.Rectangle((j, i), 1, 1, facecolor="white", edgecolor="white")
            
            if pd.notna(s_val):
                # Triangle superior (Sensibilitat): polygon de (j, i+1), (j+1, i+1), (j, i)
                poly_sens = patches.Polygon([[j, i+1], [j+1, i+1], [j, i]], facecolor=cmap_sens(norm(s_val)), edgecolor="none")
                ax.add_patch(poly_sens)
                # Text Sensibilitat (amunt a l'esquerra)
                ax.text(j + 0.28, i + 0.68, f"{s_val:.3f}", color="black", fontsize=8.5, fontweight="bold", ha="center", va="center")

            if pd.notna(e_val):
                # Triangle inferior (Especificitat): polygon de (j+1, i), (j+1, i+1), (j, i)
                poly_esp = patches.Polygon([[j+1, i], [j+1, i+1], [j, i]], facecolor=cmap_esp(norm(e_val)), edgecolor="none")
                ax.add_patch(poly_esp)
                # Text Especificitat (avall a la dreta)
                ax.text(j + 0.72, i + 0.32, f"{e_val:.3f}", color="black", fontsize=8.5, fontweight="bold", ha="center", va="center")

            # Marc de la cel·la
            cell_box = patches.Rectangle((j, i), 1, 1, facecolor="none", edgecolor="#dddddd", linewidth=1)
            ax.add_patch(cell_box)

    # Configuració dels eixos
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0) # Invertit perquès les files vagin de dalt a baix
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels(pivot_sens.columns, fontsize=10, fontweight="bold", rotation=15)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_yticklabels(pivot_sens.index, fontsize=10, fontweight="bold")

    ax.set_title("Optimism-Corrected Sensitivity (Upper-Left) & Specificity (Lower-Right)", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylabel("Model & Data Treatment", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_xlabel("Threshold Selection Criterion", fontsize=11, fontweight="bold", labelpad=10)

    # Afegir barres de color indicatives (Colorbars)
    sm_sens = plt.cm.ScalarMappable(cmap=cmap_sens, norm=norm)
    sm_sens.set_array([])
    cbar_sens = fig.colorbar(sm_sens, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar_sens.set_label('Sensitivity Scale (YlGnBu)', fontsize=9, fontweight='bold')

    sm_esp = plt.cm.ScalarMappable(cmap=cmap_esp, norm=norm)
    sm_esp.set_array([])
    cbar_esp = fig.colorbar(sm_esp, ax=ax, orientation='vertical', fraction=0.03, pad=0.08)
    cbar_esp.set_label('Specificity Scale (YlOrRd)', fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, "sensitivity_specificity_split_heatmap.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Heatmap de cel·les partides desat a: {os.path.abspath(out_png)}")

if __name__ == "__main__":
    csv_input = "bootstrap_results_table.csv"
    generar_grafic_auroc_grouped(csv_input, output_dir="results")
    generar_heatmap_partit(csv_input, output_dir="results")