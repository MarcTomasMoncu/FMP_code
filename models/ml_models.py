import pickle
from sklearn import svm, tree, naive_bayes, discriminant_analysis
from sklearn.ensemble import RandomForestClassifier # <--- AFEGIR AQUESTA LÍNIA
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from utils.metrics import calculate_metrics


def initialize_models():
    return {
        "BernoulliNB": naive_bayes.BernoulliNB(),
        "DecisionTreeClassifier": tree.DecisionTreeClassifier(max_depth=3),
        "SVC": svm.SVC(probability=True),
        "QuadraticDiscriminantAnalysis": discriminant_analysis.QuadraticDiscriminantAnalysis(reg_param=0.1),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBClassifier": XGBClassifier(learning_rate=0.1, max_depth=3)
    }

def cross_validate_model(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        y_val_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
        metrics = calculate_metrics(y_val, y_val_pred)
        metrics_list.append(metrics)
    return metrics_list

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_file):
    model.fit(X_train, y_train)
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    y_test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    metrics = calculate_metrics(y_test, y_test_pred)
    return metrics, y_test_pred