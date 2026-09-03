import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def load_dataset(file_path, exclude_columns=None, target_column="infection"):
    data = pd.read_csv(file_path, sep=",")
    if exclude_columns:
        data = data.drop(columns=exclude_columns, errors="ignore") #eliminate the columns that we do not want to use for the model
    X = data.drop(columns=[target_column]) #dataset without the target variable
    y = data[target_column].values #dataset with the target variable
    return X, y, X.columns.tolist()

def preprocess_full_dataset(file_path, exclude_columns=None, target_column="infection", random_state=42, normalize=True, apply_smote=True):
    X, y, feature_names = load_dataset(file_path, exclude_columns, target_column)
    
    if normalize:
        scaler = MinMaxScaler() #put all between 0 and 1
        X_scaled = scaler.fit_transform(X) #fit the scaler to the data and transform it
        X = pd.DataFrame(X_scaled, columns=feature_names) #convert the scaled data back to a DataFrame with the original feature names
    else:
        scaler = None

    if apply_smote:
        smote = SMOTE(random_state=random_state) #create fictitious samples of the minority class to balance the dataset
        X, y = smote.fit_resample(X, y)

    return X, y, scaler, feature_names