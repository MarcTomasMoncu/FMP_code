import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTENC, SMOTE

def load_dataset(file_path, exclude_columns=None, target_column="infection"):
    data = pd.read_csv(file_path, sep=",")
    if exclude_columns:
        data = data.drop(columns=exclude_columns, errors="ignore")
    X = data.drop(columns=[target_column])
    y = data[target_column].values
    return X, y, X.columns.tolist()

def get_categorical_indices(X):
    """
    Detecta els índexs de les columnes categòriques, booleanes o binàries (0/1).
    """
    cat_indices = []
    for i, col in enumerate(X.columns):
        unique_vals = set(X[col].dropna().unique())
        is_binary = unique_vals.issubset({0, 1, 0.0, 1.0})
        is_discrete = X[col].dtype in ['object', 'category', 'bool'] or (X[col].nunique() <= 10)
        
        if is_binary or is_discrete:
            cat_indices.append(i)
    return cat_indices

def preprocess_full_dataset(file_path, exclude_columns=None, target_column="infection", random_state=42, normalize=False, apply_smote=False):
    """
    Carrega el dataset. Per defecte normalize=False per permetre que l'escalat
    es faci estrictament dins de cada iteració del bootstrap.
    """
    X, y, feature_names = load_dataset(file_path, exclude_columns, target_column)
    cat_indices = get_categorical_indices(X)

    scaler = None
    if normalize:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_names)

    if apply_smote:
        if 0 < len(cat_indices) < X.shape[1]:
            smote = SMOTENC(categorical_features=cat_indices, random_state=random_state)
        else:
            smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(X, y)

    return X, y, scaler, feature_names