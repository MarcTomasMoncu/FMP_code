import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  

try:
    df = pd.read_csv("alertes_savac_COT_v1_enriquit.csv")
except FileNotFoundError:
    print("Error: File not found.")
    exit()

df.columns = df.columns.str.strip()

# Definition of variable types based on your dataset
vars_quantitatives = ["EDAT", "ESTADA_TOTAL", "ESTADA_POSTOPERATORIA", "DIES_ENTRE_FI_OPERACIO_I_DEFUNCIO"]
vars_categoriques = [
    "ANTIBIOTIC", "MICROBIOLOGIA", "MORT_HOSPITALARIA", "MORT_POSTERIORMENT", 
    "REINGRES_URG", "REINTERVENCIO", "REFERENT_ACTUAL", "SEXE"
]

# Clean variables that might not be present in the dataframe
vars_categoriques = [v for v in vars_categoriques if v in df.columns]

# ==============================================================================
# 1. GENERATION OF THE SPECIALIZED GENERAL DESCRIPTIVE TABLE
# ==============================================================================
rows_general = []

# Metrics for quantitative variables (Mean ± SD and Median)
for v in vars_quantitatives:
    if v in df.columns:
        desc = df[v].describe()
        rows_general.append({
            "Variable": f"{v}",
            "Statistics": f"Mean: {desc['mean']:.2f} ± {desc['std']:.2f} (Median: {desc['50%']:.0f})"
        })

# Metrics for categorical/binary variables (Counts of active cases = 1 and percentages)
for v in vars_categoriques:
    total_valids = df[v].dropna().count()
    casos_actius = (df[v] == 1).sum()
    pct = (casos_actius / total_valids) * 100 if total_valids > 0 else 0
    rows_general.append({
        "Variable": f"{v}",
        "Statistics": f"Cases: {casos_actius} of {total_valids} ({pct:.1f}%)"
    })

taula_general_exp = pd.DataFrame(rows_general)

# ==============================================================================
# 2. GENERATION OF THE COMPARATIVE TABLE BY INFECTION STATUS
# ==============================================================================
rows_infeccio = []

df_0 = df[df["INFECCIO"] == 0]
df_1 = df[df["INFECCIO"] == 1]

# Comparison for quantitative variables (Mean ± SD)
for v in vars_quantitatives:
    if v in df.columns:
        m0, s0 = df_0[v].mean(), df_0[v].std()
        m1, s1 = df_1[v].mean(), df_1[v].std()
        rows_infeccio.append({
            "Variable": v,
            "No Infection (0)": f"{m0:.2f} ± {s0:.2f}" if not pd.isna(m0) else "N/A",
            "Infection (1)": f"{m1:.2f} ± {s1:.2f}" if not pd.isna(m1) else "N/A"
        })

# Comparison for categorical variables (Frequency and Percentage of active cases)
for v in vars_categoriques:
    n0 = df_0[v].dropna().count()
    c0 = (df_0[v] == 1).sum()
    p0 = (c0 / n0) * 100 if n0 > 0 else 0
    
    n1 = df_1[v].dropna().count()
    c1 = (df_1[v] == 1).sum()
    p1 = (c1 / n1) * 100 if n1 > 0 else 0
    
    rows_infeccio.append({
        "Variable": v,
        "No Infection (0)": f"{c0}/{n0} ({p0:.1f}%)",
        "Infection (1)": f"{c1}/{n1} ({p1:.1f}%)"
    })

taula_infeccio_exp = pd.DataFrame(rows_infeccio)

# ==============================================================================
# 3. FUNCTION TO EXPORT FORMATTED TABLES TO PNG
# ==============================================================================
def exportar_a_png_millorat(df_table, nom_fitxer, titol):
    fig, ax = plt.subplots(figsize=(14, len(df_table) * 0.4 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(titol, fontweight="bold", fontsize=15, pad=15, color="#1a365d")
    
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    
    # Clean and professional styling
    for (row, col), cell in table.get_celld().items():
        cell.set_fontsize(10)
        cell.set_edgecolor("#e2e8f0")
        if row == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and row % 2 == 0:
            cell.set_facecolor("#f7fafc")
            
    table.scale(1.0, 1.5)
    
    full_path = os.path.join(output_dir, nom_fitxer)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

# Exporting final results
exportar_a_png_millorat(taula_general_exp, "taula_general_explicativa.png", "Dataset Descriptive Analysis")
exportar_a_png_millorat(taula_infeccio_exp, "taula_infeccio_explicativa.png", "Clinical Profile by Infection Status")

print("Process completed successfully! The new specialized tables have been saved in the 'results/' folder.")

# ==============================================================================
# ADDITIONAL ANALYSIS: 2x2 CONTINGENCY TABLE (CONFUSION MATRIX)
# ==============================================================================

# Cross-tabulate INFECCIO (True Status) vs REFERENT_ACTUAL (Current Model Model)
contingency_matrix = pd.crosstab(
    df["REFERENT_ACTUAL"], 
    df["INFECCIO"],
    dropna=False
)

# Rename indices and columns for better clarity in the English final report
contingency_matrix.index = ["Current Model: Negative (0)", "Current Model: Positive (1)"]
contingency_matrix.columns = ["True Status: No Infection (0)", "True Status: Infection (1)"]

# Specialized function to plot the 2x2 matrix with a professional design
def export_contingency_2x2(df_table, nom_fitxer, titol):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(titol, fontweight="bold", fontsize=14, pad=20, color="#1a365d")
    
    table_data = df_table.reset_index()
    
    table = ax.table(
        cellText=table_data.values, 
        colLabels=[""] + list(df_table.columns), 
        loc='center', 
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_fontsize(11)
        cell.set_edgecolor("#cbd5e1")
        
        # Header row styling (Columns)
        if row == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", fontweight="bold")
        # First column styling (Row Labels)
        elif col == 0 and row > 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(fontweight="bold", color="#2d3748")
        # Internal 2x2 data cells
        elif row > 0 and col > 0:
            cell.set_facecolor("#ffffff")
            cell.set_text_props(fontweight="medium")
            
    table.scale(1.2, 2.0)
    
    full_path = os.path.join(output_dir, nom_fitxer)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

# Export the new 2x2 matrix image
export_contingency_2x2(
    contingency_matrix, 
    "contingency_matrix_2x2.png", 
    "2x2 Contingency Table: True Infection vs Current Screening Model"
)

print("2x2 contingency table successfully generated at 'results/contingency_matrix_2x2.png'.")