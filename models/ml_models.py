import pickle
from sklearn import svm, tree, naive_bayes, discriminant_analysis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from utils.metrics import calculate_metrics


def initialize_models(): #function to initialize the machine learning models that will be trained and evaluated in the pipeline, returning a dictionary with the model names as keys and the initialized model objects as values
    return {
        #"BernoulliNB": naive_bayes.BernoulliNB(), 
        #"DecisionTreeClassifier": tree.DecisionTreeClassifier(max_depth=3), 
        #"SVC": svm.SVC(probability=True),
        "QuadraticDiscriminantAnalysis": discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param=0.1),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBClassifier": XGBClassifier(learning_rate=0.1, max_depth=3)
    }

def cross_validate_model(model, X, y, n_splits=5): 
    # Mantenim la proporció exacta de positius/negatius a cada fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []
    
    for train_index, val_index in skf.split(X, y): 
        # Separem les dades assegurant compatibilitat si X és DataFrame o Numpy Array
        X_train_fold = X.values[train_index] if hasattr(X, "values") else X[train_index]
        X_val_fold = X.values[val_index] if hasattr(X, "values") else X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # apliquem SMOTE NOMÉS al subgrup d'entrenament d'aquest fold específic
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_fold, y_train_fold)
        
        # Entrenem el model amb les dades balancejades del fold
        model.fit(X_train_res, y_train_res)
        
        # Avaluem sobre dades de validació REALS (sense dades sintètiques del SMOTE)
        y_val_pred = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val_fold)
        metrics = calculate_metrics(y_val_fold, y_val_pred)
        metrics_list.append(metrics)
        
    return metrics_list

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_file): #function to train a given machine learning model on the training data, save the trained model to a file and evaluate it on the test data, returning the calculated metrics and predicted probabilities for the test set
    model.fit(X_train, y_train)
    with open(model_file, 'wb') as f: #save the trained model to a file using pickle
        pickle.dump(model, f)
    y_test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    metrics = calculate_metrics(y_test, y_test_pred)
    return metrics, y_test_pred