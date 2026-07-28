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

def generar_grafic_auroc_grouped(csv_path="taula_resultats_bootstrap.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El fitxer '{csv_path}' no existeix. Comprova la ruta.")

    print("Carregant dades del CSV...")
    df = pd.read_csv(csv_path)

    # 1. Filtrar per un criteri representatiu (Youden) per evitar duplicar entrades per model/tractament
    df_youden = df[df["Criteri_Llindar"] == "Youden"].copy()
    if df_youden.empty:
        df_youden = df.drop_duplicates(subset=["Model", "Tractament"]).copy()

    # 2. Traduir i estandarditzar els noms per a la publicació en anglès
    model_map = {
        "RegressioLogisticaPenalitzada": "Penalized Logistic Reg.",
        "QuadraticDiscriminantAnalysis": "QDA",
        "RandomForestClassifier": "Random Forest",
        "XGBClassifier": "XGBoost",
        "DenseNeuralNet": "DNN"
    }

    treatment_map = {
        "Sense_Tractament": "No Treatment",
        "Ponderacio": "Class Weighting",
        "SMOTE": "SMOTENC",
        "SMOTENC": "SMOTENC"
    }

    df_youden["Model_EN"] = df_youden["Model"].map(lambda x: model_map.get(x, x))
    df_youden["Tractament_EN"] = df_youden["Tractament"].map(lambda x: treatment_map.get(x, x))

    # Parsejar valors d'AUROC i intervals de confiança
    parsed_data = []
    for idx, row in df_youden.iterrows():
        val, err_inf, err_sup = extreure_ic(row["AUROC (IC 95%)"])
        if val is None:
            val = row["AUROC_Corregit"]
            err_inf, err_sup = 0, 0
            
        parsed_data.append({
            "Model": row["Model_EN"],
            "Treatment": row["Tractament_EN"],
            "AUROC": val,
            "Err_Inf": err_inf,
            "Err_Sup": err_sup
        })

    df_plot = pd.DataFrame(parsed_data)

    # 3. Ordre específic sol·licitat per als models
    models_order = [
        "Penalized Logistic Reg.",
        "QDA",
        "Random Forest",
        "XGBoost",
        "DNN"
    ]
    
    treatments = ["No Treatment", "Class Weighting", "SMOTENC"]

    # 4. Configuració del gràfic de barres agrupades
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Paleta de colors
    palette = {
        "No Treatment": "#e74c3c",       # Vermell
        "Class Weighting": "#2ecc71",    # Verd
        "SMOTENC": "#3498db"             # Blau
    }

    x = np.arange(len(models_order))
    width = 0.25  # Ample de les barres

    # 5. Dibuixar les barres per a cada tractament seguint l'ordre establert
    for i, tr in enumerate(treatments):
        df_tr = df_plot[df_plot["Treatment"] == tr]
        
        vals = []
        errs_inf = []
        errs_sup = []
        
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
            x + offset, 
            vals, 
            width, 
            yerr=[errs_inf, errs_sup],
            capsize=4,
            label=tr, 
            color=palette.get(tr, "#95a5a6"),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85
        )

        # Afegir etiquetes amb el valor exacte sobre les barres
        for rect, val in zip(rects, vals):
            if val > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    val + 0.025,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold"
                )

    # 6. Personalització del gràfic (Sense títol)
    ax.set_ylabel("Optimism-Corrected AUROC (95% CI)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models_order, fontsize=10, fontweight="bold")
    ax.set_ylim([0.4, 1.02])
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random Chance (0.50)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Llegenda
    ax.legend(title="Data Treatment", title_fontsize='10', loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()

    # 7. Desar el resultat
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, "auroc_grouped_by_treatment.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gràfic generat amb èxit i desat a: {os.path.abspath(out_png)}")

if __name__ == "__main__":
    generar_grafic_auroc_grouped("taula_resultats_bootstrap.csv", output_dir="results")