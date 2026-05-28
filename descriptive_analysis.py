import os
import pandas as pd
import matplotlib.pyplot as plt
from weasyprint import HTML

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  


try:
    df = pd.read_csv("fake_dataset_v2.csv")
except FileNotFoundError:
    print("Error: No s'ha trobat el fitxer 'fake_dataset_v2.csv'.")
    exit()

df.columns = df.columns.str.strip()

taula_general = df.describe().T.round(2)[["count", "mean", "std", "min", "50%", "max"]]
taula_per_infeccio = df.groupby("infection").mean().T.round(2)
taula_per_infeccio.columns = ["Sense Infecció (0)", "Amb Infecció (1)"]


def exportar_a_png(df_table, nom_fitxer, titol):
    fig, ax = plt.subplots(figsize=(12, len(df_table) * 0.35 + 1.5))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(titol, fontweight="bold", fontsize=14, pad=10, color="#1a365d")
    
    table_data = df_table.reset_index()
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_fontsize(9)
        if row == 0:
            cell.set_facecolor("#2b6cb0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and row % 2 == 0:
            cell.set_facecolor("#f7fafc")
            
    table.scale(1.0, 1.4)
    
    full_path = os.path.join(output_dir, nom_fitxer)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()

exportar_a_png(taula_general, "taula_general.png", "Taula Descriptiva General")
exportar_a_png(taula_per_infeccio, "taula_infeccio.png", "Mitjanes Comparatives segons Infecció")
