
PROJECTE TFM: Vigilància d'Infeccions de Lloc Quirúrgic (SSI) en Trauma


Aquest projecte implementa un sistema de triatge automatitzat per a la 
identificació retrospectiva d'infeccions de lloc quirúrgic (SSI) en la 
especialitat de cirurgia de trauma. Compara models estadístics clàssics, 
Machine Learning i Deep Learning.

------------------------------------------------------------------------
1. ESTRUCTURA DEL PROJECTE
------------------------------------------------------------------------

L'arquitectura està modularitzada per facilitar la reproductibilitat:

- config.json: Fitxer central de configuració. Aquí es defineixen les 
  variables, les rutes de dades i el llindar de sensibilitat (90%).
- train_pipeline.py: Script principal que executa tot el procés: càrrega, 
  entrenament, validació creuada, avaluació i interpretabilitat.
- preprocessing.py: Gestiona la càrrega de dades, la normalització, 
  l'estratificació (stratify) i el balanceig de classes (SMOTE).
- models/
    - ml_models.py: Definició dels models de Machine Learning (RF, XGB, 
      QDA, etc.) i la seva validació creuada.
    - dl_models.py: Definició de la Xarxa Neural Densa (DNN) amb TensorFlow.
- utils/
    - metrics.py: Càlcul de mètriques clíniques (Sensibilitat, Especificitat, 
      VPP, VPN, AUC) i cerca dinàmica del llindar (threshold).
- interpretability.py: Generació d'explicacions clíniques mitjançant SHAP.
- .gitignore: Filtre per no pujar fitxers innecessaris (entorns virtuals, 
  dades privades, etc.) al repositori.

------------------------------------------------------------------------
2. PREPARACIÓ DE L'ENTORN
------------------------------------------------------------------------

Passos per configurar l'ordinador abans d'executar el codi:

1. Crear l'entorn virtual:
   $ python3 -m venv tfm_env

2. Activar l'entorn:
   $ source tfm_env/bin/activate

3. Instal·lar les llibreries necessàries:
   $ pip install xgboost shap imbalanced-learn scikit-learn pandas numpy tensorflow matplotlib

------------------------------------------------------------------------
3. EXECUCIÓ DELS MODELS
------------------------------------------------------------------------

Una vegada configurat l'entorn i preparat el fitxer de dades (ex: dades.csv):

1. Configurar config.json:
   - Modifica "dataset_path" amb el nom del teu fitxer.
   - Revisa "target_column" (la variable a predir).
   - Especifica "exclude_columns" (IDs de pacients o dades no predictives).

2. Executar el pipeline:
   $ python3 train_pipeline.py

------------------------------------------------------------------------
4. RESULTATS I OUTPUTS
------------------------------------------------------------------------

El sistema generarà automàticament dues carpetes:

- results/: 
    - performance_results.csv: Taula comparativa de tots els models amb 
      el llindar ajustat al 90% de sensibilitat.
    - cross_validation_results.csv: Resultats de la robustesa dels models.
    - prediction_probs.csv: Probabilitats predites per a cada pacient.
    - Gràfics SHAP (.png): Visualització de quines variables afecten més al risc.

- artifacts/:
    - Guarda els models ja entrenats en format .pkl (ML) o .h5 (DL) per 
      poder-los utilitzar en el futur sense tornar a entrenar.

------------------------------------------------------------------------
Notes: Aquest codi està dissenyat com a eina de suport a la decisió clínica. 
Sempre s'ha de mantenir la supervisió humana per a la validació final 
dels casos d'infecció.
