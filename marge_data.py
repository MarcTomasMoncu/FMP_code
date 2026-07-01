import pandas as pd

# 1. Carregar els fitxers CSV
df_alertes = pd.read_csv("alertes_savac_COT_v1.csv")
df_base_final = pd.read_csv("base_final_limpia_1619.csv")

# 2. Seleccionar les columnes d'interès del fitxer base_final
columnes_interes = ["nhc", "ilq", "pt_edad", "pt_sexo", "pt_desc_sexo_h"]
df_base_reduit = df_base_final[columnes_interes]

# 3. Netejar duplicats de la base de dades clínica pel TFM
df_base_reduit = df_base_reduit.drop_duplicates(subset=["nhc"], keep="first")

# 4. Assegurar que els IDs siguin Strings
df_alertes["CI22NUMHISTORIA"] = df_alertes["CI22NUMHISTORIA"].astype(str)
df_base_reduit["nhc"] = df_base_reduit["nhc"].astype(str)

# 5. Fer el merge
df_resultat = pd.merge(
    df_alertes,
    df_base_reduit,
    left_on="CI22NUMHISTORIA",
    right_on="nhc",
    how="left"
)
print(df_resultat.head())

df_resultat = df_resultat.drop(columns=["nhc"])

print(df_resultat.head())

# ==============================================================================
# 6. TRADUCCIÓ A 0 I 1 (BINARITZACIÓ)
# ==============================================================================

# Llista de columnes d'alertes basades en SI/NO
columnes_sino = [
    "ALERTAANTIBIOTIC", "ALERTAMICROBIOLOGIA", "EXITUSINTRAHOSPITALARI", 
    "ESMORTPOSTERIORMENT", "ALERTAREINGRESURG", "ALERTAREINTERVENCIO", "ESALERTA"
]

# Reemplacem SI -> 1 i NO -> 0
for col in columnes_sino:
    if col in df_resultat.columns:
        df_resultat[col] = df_resultat[col].replace({"SI": 1, "NO": 0})

# Tractament especial per a 'ilq'
if "ilq" in df_resultat.columns:
    df_resultat["ilq"] = df_resultat["ilq"].astype(str).str.strip()
    df_resultat["ilq"] = df_resultat["ilq"].replace({"NO": 0})
    df_resultat["ilq"] = df_resultat["ilq"].replace(to_replace=r'.*ILQ.*', value=1, regex=True)

# ==============================================================================
# 7. MILLORA: TREURE ELS DECIMALS (.0) D'EDAT I SEXE
# ==============================================================================
# Convertim a "Int64" (admet valors buits sense transformar-los a decimals)
if "pt_edad" in df_resultat.columns:
    df_resultat["pt_edad"] = df_resultat["pt_edad"].astype("Int64")

if "pt_sexo" in df_resultat.columns:
    df_resultat["pt_sexo"] = df_resultat["pt_sexo"].astype("Int64")
# ==============================================================================

# 8. Guardar el resultat final
df_resultat.to_csv("alertes_savac_COT_v1_enriquit.csv", index=False)

print(df_resultat.head())
print("Procés completat correctament! Dades guardades, binaritzades i sense decimals decimals.")