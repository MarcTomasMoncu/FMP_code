import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_dataset(file_path, exclude_columns=None, target_column="infection"):
    data = pd.read_csv(file_path, sep=",")
    if exclude_columns:
        data = data.drop(columns=exclude_columns, errors="ignore") #eliminate the columns that we do not want to use for the model
    X = data.drop(columns=[target_column]) #dataset without the target variable
    y = data[target_column].values #dataset with the target variable
    return X, y, X.columns.tolist()

def split_and_preprocess(file_path, exclude_columns=None, target_column="infection", test_size=0.2, random_state=42, normalize=True, apply_smote=True):
    X, y, feature_names = load_dataset(file_path, exclude_columns, target_column)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) #divide the dataset into training and testing sets, maintaining the same proportion of classes in both sets (stratify=y) --> to be sure that we have 5% infections

    if normalize:
        scaler = MinMaxScaler() #put all between 0 and 1
        X_train_scaled = scaler.fit_transform(X_train)#fit the scaler to the data and transform it
        X_test_scaled = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train_scaled, columns=feature_names)#convert the scaled data back to a DataFrame with the original feature names
        X_test = pd.DataFrame(X_test_scaled, columns=feature_names)
    else:
        scaler = None

    if apply_smote:
        smote = SMOTE(random_state=random_state)#create fictitious samples of the minority class (infections) to balance the dataset
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler, feature_names