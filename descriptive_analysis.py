import pandas as pd
import matplotlib.pyplot as plt
from weasyprint import HTML


try:
    df = pd.read_csv("fake_dataset_v2.csv")
except FileNotFoundError:
    print("Error: No s'ha trobat el fitxer 'fake_dataset_v2.csv'.")
    exit()

df.columns = df.columns.str.strip()


taula_general = df.describe().T.round(2)[["count", "mean", "std", "min", "50%", "max"]]
taula_per_infeccio = df.groupby("infection").mean().T.round(2)
taula_per_infeccio.columns = ["Sense Infecció (0)", "Amb Infecció (1)"]


html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{ size: A4; margin: 20mm 15mm; }}
    body {{ font-family: Arial, sans-serif; color: #2c3e50; font-size: 10pt; }}
    h1 {{ color: #1a365d; border-bottom: 2px solid #2b6cb0; padding-bottom: 5px; }}
    h2 {{ color: #2b6cb0; margin-top: 25px; border-left: 4px solid #2b6cb0; padding-left: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
    th {{ background-color: #2b6cb0; color: white; padding: 8px; text-align: left; font-size: 9.5pt; }}
    td {{ padding: 6px 8px; border: 1px solid #e2e8f0; }}
    tr:nth-child(even) {{ background-color: #f7fafc; }}
</style>
</head>
<body>
    <h1>Informe Descriptiu Clínic</h1>
    
    <h2>1. Taula Descriptiva General</h2>
    <table>
        <tr>
            <th>Variable</th><th>Recompte</th><th>Mitjana</th><th>Desv. Est.</th><th>Mínim</th><th>Mediana</th><th>Màxim</th>
        </tr>
"""
for idx, row in taula_general.iterrows():
    html_content += f"<tr><td><strong>{idx}</strong></td><td>{row['count']}</td><td>{row['mean']}</td><td>{row['std']}</td><td>{row['min']}</td><td>{row['50%']}</td><td>{row['max']}</td></tr>"

html_content += """
    </table>
    <h2>2. Mitjanes Comparatives segons Infecció</h2>
    <table>
        <tr><th>Variable</th><th>Sense Infecció (0)</th><th>Amb Infecció (1)</th></tr>
"""
for idx, row in taula_per_infeccio.iterrows():
    html_content += f"<tr><td><strong>{idx}</strong></td><td>{row['Sense Infecció (0)']}</td><td>{row['Amb Infecció (1)']}</td></tr>"

html_content += "</table></body></html>"


with open("informe_temporal.html", "w", encoding="utf-8") as f:
    f.write(html_content)
HTML("informe_temporal.html").write_pdf("informe_descriptiu.pdf")


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
    plt.savefig(nom_fitxer, dpi=300, bbox_inches='tight')
    plt.close()


exportar_a_png(taula_general, "taula_general.png", "Taula Descriptiva General")
exportar_a_png(taula_per_infeccio, "taula_infeccio.png", "Mitjanes Comparatives segons Infecció")
