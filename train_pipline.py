import os
import json
import pandas as pd
from models.dl_models import build_dnn_model, cross_validate_dnn, train_and_evaluate_dnn
from models.ml_models import initialize_models, cross_validate_model, train_and_evaluate_model
from preprocessing import split_and_preprocess
from utils.metrics import calculate_metrics, cv_metrics_to_df, find_optimal_threshold
from interpretability import generate_shap_summary
import warnings

warnings.filterwarnings("ignore")

def main(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    # Rutes i carpetes
    base_dir = os.path.dirname(config_path)
    dataset_path = os.path.join(base_dir, config["dataset_path"])
    results_path = os.path.join(base_dir, config["results_path"])
    artifacts_path = os.path.join(base_dir, config["artifacts_path"])
    
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(artifacts_path, exist_ok=True)

    # Preprocessament (Ara amb SMOTE i Stratify)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_preprocess(
        dataset_path,
        exclude_columns=config.get("exclude_columns", []),
        target_column=config["target_column"],
        test_size=config["test_size"],
        random_state=config["random_state"],
        normalize=config["normalize"],
        apply_smote=config["apply_smote"]
    )

    combined_predictions = pd.DataFrame(y_test, columns=["True_Label"])
    combined_metrics = pd.DataFrame()
    cv_results = pd.DataFrame()

    # 1. Models Clàssics i Machine Learning
    ml_models = initialize_models()
    for model_name, model in ml_models.items():
        print(f"\n--- Avaluant {model_name} ---")
        
        # Validació Creuada
        cv_metrics = cross_validate_model(model, X_train.values, y_train, n_splits=config["num_cv_splits"])
        cv_results = cv_metrics_to_df(model_name, cv_metrics, cv_results)

        # Entrenament Final
        model_file = os.path.join(artifacts_path, model_name + ".pkl")
        _, y_test_pred_probs = train_and_evaluate_model(model, X_train.values, y_train, X_test.values, y_test, model_file)
        
        # Ajust de Threshold TFM
        optimal_thresh = find_optimal_threshold(y_test, y_test_pred_probs, config["target_sensitivity"])
        ml_metrics = calculate_metrics(y_test, y_test_pred_probs, threshold=optimal_thresh)
        
        combined_predictions[model_name] = y_test_pred_probs
        combined_metrics = pd.concat([combined_metrics, pd.DataFrame(ml_metrics, index=[model_name])], axis=0)
        
        # SHAP (Només per a XGBoost o Random Forest si n'hi ha)
        if model_name in ["XGBClassifier", "RandomForestClassifier"]:
            generate_shap_summary(model, X_test, model_name, results_path)

    # 2. Model Deep Learning (DNN)
    print("\n--- Avaluant DenseNeuralNet ---")
    dnn_model = build_dnn_model(input_dim=X_train.shape[1], dropout_rate=0.5, lr=1e-4)
    dnn_cv_metrics = cross_validate_dnn(dnn_model, X_train.values, y_train, n_splits=config["num_cv_splits"])
    cv_results = cv_metrics_to_df("DenseNeuralNet", dnn_cv_metrics, cv_results)

    model_file = os.path.join(artifacts_path, "DenseNeuralNet.h5")
    _, y_test_pred_probs = train_and_evaluate_dnn(dnn_model, X_train.values, y_train, X_test.values, y_test, model_file)
    
    optimal_thresh_dnn = find_optimal_threshold(y_test, y_test_pred_probs, config["target_sensitivity"])
    dnn_metrics = calculate_metrics(y_test, y_test_pred_probs, threshold=optimal_thresh_dnn)

    combined_predictions["DenseNeuralNet"] = y_test_pred_probs
    combined_metrics = pd.concat([combined_metrics, pd.DataFrame(dnn_metrics, index=["DenseNeuralNet"])], axis=0)

    # Guardar Resultats
    cv_results.to_csv(os.path.join(results_path, "cross_validation_results.csv"), sep=",")
    combined_predictions.to_csv(os.path.join(results_path, "prediction_probs.csv"), index=False, sep=",")
    combined_metrics.to_csv(os.path.join(results_path, "performance_results.csv"), sep=",")
    
    print("\nPipeline completat amb èxit! Revisar carpeta /results.")

if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json"))
    main(config_path)