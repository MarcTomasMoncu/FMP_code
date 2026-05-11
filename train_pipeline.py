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
    with open(config_path, "r") as f: #read the configuration file to get the paths and parameters for the training pipeline
        config = json.load(f)

    #routes construction and folder creations
    base_dir = os.path.dirname(config_path)
    dataset_path = os.path.join(base_dir, config["dataset_path"])
    results_path = os.path.join(base_dir, config["results_path"])
    artifacts_path = os.path.join(base_dir, config["artifacts_path"])
    
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(artifacts_path, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_preprocess( #call the script of processing 
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

    ml_models = initialize_models()
    for model_name, model in ml_models.items(): #a for to start the training for all the models defined in ml_models.py
        print(f"\nTraining for the following model: {model_name} ")
        
        cv_metrics = cross_validate_model(model, X_train.values, y_train, n_splits=config["num_cv_splits"]) #Cross-validation with 5 splits to be sure of the robustness.
        cv_results = cv_metrics_to_df(model_name, cv_metrics, cv_results) #store nfo in the scv

        model_file = os.path.join(artifacts_path, model_name + ".pkl") #save the model in artifacts folder
        _, y_test_pred_probs = train_and_evaluate_model(model, X_train.values, y_train, X_test.values, y_test, model_file) #traind and evaluate the model
        
        optimal_thresh = find_optimal_threshold(y_test, y_test_pred_probs, config["target_sensitivity"]) #find optimal threshold
        ml_metrics = calculate_metrics(y_test, y_test_pred_probs, threshold=optimal_thresh) #calculate the metrics with the threshold calculated before
        
        combined_predictions[model_name] = y_test_pred_probs #store information
        combined_metrics = pd.concat([combined_metrics, pd.DataFrame(ml_metrics, index=[model_name])], axis=0) #store the metrics in a dataframe
        
        if model_name in ["XGBClassifier", "RandomForestClassifier"]: #generate the SHAP summary plot only for the models that work with trees  
            generate_shap_summary(model, X_test, model_name, results_path)
        print(f"Finished training for {model_name} model")

    print("\n--- Training DenseNeuralNet ---")
    dnn_model = build_dnn_model(input_dim=X_train.shape[1], dropout_rate=0.5, lr=1e-4) #build the DNN model with the parameters defined in dl_models.py
    dnn_cv_metrics = cross_validate_dnn(dnn_model, X_train.values, y_train, n_splits=config["num_cv_splits"]) #cross-validation with 5 splits
    cv_results = cv_metrics_to_df("DenseNeuralNet", dnn_cv_metrics, cv_results) #store the cross-validation results in a dataframe

    model_file = os.path.join(artifacts_path, "DenseNeuralNet.keras") #save the DNN model in artifacts folder
    _, y_test_pred_probs = train_and_evaluate_dnn(dnn_model, X_train.values, y_train, X_test.values, y_test, model_file) #train and evaluate the DNN model
    
    optimal_thresh_dnn = find_optimal_threshold(y_test, y_test_pred_probs, config["target_sensitivity"]) #find the optimal threshold for the DNN model
    dnn_metrics = calculate_metrics(y_test, y_test_pred_probs, threshold=optimal_thresh_dnn) #calculate the metrics for the DNN model with the threshold that we calculated before

    combined_predictions["DenseNeuralNet"] = y_test_pred_probs #store the predicted probabilities of the DNN model in the combined_predictions dataframe
    combined_metrics = pd.concat([combined_metrics, pd.DataFrame(dnn_metrics, index=["DenseNeuralNet"])], axis=0) #store metrics

    cv_results.to_csv(os.path.join(results_path, "cross_validation_results.csv"), sep=",") #save stored info in the csv
    combined_predictions.to_csv(os.path.join(results_path, "prediction_probs.csv"), index=False, sep=",") 
    combined_metrics.to_csv(os.path.join(results_path, "performance_results.csv"), sep=",")  
      
if __name__ == "__main__": #execute the main function with the path of the configuration file as an argument
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.json")) 
    main(config_path)