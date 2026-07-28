import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_ci_values(text_ci):
    """
    Auxiliary function to extract corrected value and 95% CI bounds
    from strings formatted as "0.8520 (0.8120 - 0.8910)".
    """
    if pd.isna(text_ci):
        return None, None, None
    
    pattern = r"([-\d\.]+)\s*\(([-\d\.]+)\s*-\s*([-\d\.]+)\)"
    match = re.search(pattern, str(text_ci))
    
    if match:
        val = float(match.group(1))
        ci_inf = float(match.group(2))
        ci_sup = float(match.group(3))
        return val, ci_inf, ci_sup
    return None, None, None

def generate_combined_plots(csv_path="taula_resultats_bootstrap.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found. Please run the Bootstrap script first.")
        return

    # Ensure results directory exists
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset from CSV...")
    df = pd.read_csv(csv_path)

    # -------------------------------------------------------------------------
    # 1. DATA PROCESSING FOR TRADEOFF AND HEATMAP
    # -------------------------------------------------------------------------
    data_tradeoff = []
    
    for idx, row in df.iterrows():
        sens_val, sens_inf, sens_sup = extract_ci_values(row.get("Sensibilitat (IC 95%)"))
        spec_val, spec_inf, spec_sup = extract_ci_values(row.get("Especificitat (IC 95%)"))
        
        if sens_val is not None and spec_val is not None:
            data_tradeoff.append({
                "Model": row["Model"],
                "Treatment": row["Tractament"],
                "Threshold_Criterion": row["Criteri_Llindar"],
                "Sensitivity": sens_val,
                "Sens_Err_Inf": sens_val - sens_inf,
                "Sens_Err_Sup": sens_sup - sens_val,
                "Specificity": spec_val,
                "Spec_Err_Inf": spec_val - spec_inf,
                "Spec_Err_Sup": spec_sup - spec_val
            })

    df_clean = pd.DataFrame(data_tradeoff)

    # Prepare data for Heatmap (Filtering by representative criterion, e.g., Youden)
    df_heatmap_raw = df[df["Criteri_Llindar"] == "Youden"].copy()
    
    # Create combined row label (Model + Data Treatment)
    df_heatmap_raw["Model_Treatment"] = df_heatmap_raw["Model"] + " | " + df_heatmap_raw["Tractament"]
    
    cols_impact = [
        "Pacients_Evitats_Corregit", 
        "Pct_Evitats_Corregit", 
        "Pacients_Marcats_Corregit", 
        "Pct_Marcats_Corregit", 
        "Infeccions_Perdudes_Corregit"
    ]
    
    heatmap_matrix = df_heatmap_raw.set_index("Model_Treatment")[cols_impact]
    
    # English Column Headers for Heatmap
    heatmap_matrix.columns = [
        "Avoided Reviews (N)", 
        "% Avoided", 
        "Flagged Patients (N)", 
        "% Flagged", 
        "Missed Infections (N)"
    ]

    # -------------------------------------------------------------------------
    # 2. FIGURE CONFIGURATION (2 VERTICAL SUBPLOTS)
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, figsize=(13, 16))

    # =========================================================================
    # TOP SUBPLOT: CLINICAL TRADEOFF (SENSITIVITY VS SPECIFICITY)
    # =========================================================================
    palette_treatments = {
        "Sense_Tractament": "#e74c3c",  # Red
        "Ponderacio": "#2ecc71",        # Green
        "SMOTENC": "#3498db"            # Blue
    }

    markers_models = {
        "RegressioLogisticaPenalitzada": "o",
        "QuadraticDiscriminantAnalysis": "s",
        "RandomForestClassifier": "^",
        "XGBClassifier": "D",
        "DenseNeuralNet": "X"
    }

    for idx, row in df_clean.iterrows():
        tr = row["Treatment"]
        model = row["Model"]
        color = palette_treatments.get(tr, "#7f8c8d")
        marker = markers_models.get(model, "o")

        # 95% CI Error bars
        ax_top.errorbar(
            x=row["Specificity"],
            y=row["Sensitivity"],
            xerr=[[row["Spec_Err_Inf"]], [row["Spec_Err_Sup"]]],
            yerr=[[row["Sens_Err_Inf"]], [row["Sens_Err_Sup"]]],
            fmt='none',
            ecolor=color,
            alpha=0.35,
            capsize=3,
            elinewidth=1.2
        )

        # Main Data Point
        ax_top.scatter(
            row["Specificity"],
            row["Sensitivity"],
            color=color,
            marker=marker,
            s=85,
            edgecolor='black',
            linewidth=0.7,
            zorder=5
        )

    ax_top.set_title(
        "A. Clinical Trade-Off: Sensitivity vs. Specificity by Data Treatment\n(Bootstrap-Adjusted with 95% CI)",
        fontsize=13, fontweight="bold", pad=12, color="#1a365d"
    )
    ax_top.set_xlabel("Corrected Specificity (1 - False Positive Rate)", fontsize=10, fontweight="bold")
    ax_top.set_ylabel("Corrected Sensitivity (True Positive Rate)", fontsize=10, fontweight="bold")
    ax_top.set_xlim([0.0, 1.02])
    ax_top.set_ylim([0.0, 1.02])
    ax_top.axhline(0.80, color='gray', linestyle='--', alpha=0.6)

    # Treatments Legend
    leg_treatments = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=9, label=tr)
        for tr, col in palette_treatments.items()
    ]
    leg1 = ax_top.legend(
        handles=leg_treatments, title="Data Treatment Method", 
        loc="lower left", frameon=True, facecolor='white', framealpha=0.9
    )
    ax_top.add_artist(leg1)

    # Models Legend
    leg_models = [
        plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=mod)
        for mod, m in markers_models.items()
    ]
    ax_top.legend(
        handles=leg_models, title="Machine Learning Model", 
        loc="upper left", frameon=True, facecolor='white', framealpha=0.9
    )

    # =========================================================================
    # BOTTOM SUBPLOT: CLINICAL IMPACT HEATMAP
    # =========================================================================
    sns.heatmap(
        heatmap_matrix, 
        annot=True, 
        fmt=".1f", 
        cmap="YlGnBu", 
        linewidths=1, 
        cbar_kws={'label': 'Volume / Workload Intensity'},
        ax=ax_bottom
    )

    ax_bottom.set_title(
        "B. Clinical Impact Heatmap (Bootstrap-Adjusted - Youden Criterion)\nReview Workload Burden vs. Missed Infection Risk",
        fontsize=13, fontweight="bold", pad=12, color="#1a365d"
    )
    ax_bottom.set_xlabel("Impact Metrics & Operational Efficiency", fontsize=10, fontweight="bold")
    ax_bottom.set_ylabel("ML Model | Data Treatment Method", fontsize=10, fontweight="bold")
    ax_bottom.set_yticklabels(ax_bottom.get_yticklabels(), rotation=0)

    # Adjust layout spacing
    plt.tight_layout(pad=3.0)
    
    # Save image to results folder
    output_png = os.path.join(output_dir, "analysis_tradeoff_and_heatmap.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Combined plot generated successfully and saved to: {os.path.abspath(output_png)}")

if __name__ == "__main__":
    generate_combined_plots()