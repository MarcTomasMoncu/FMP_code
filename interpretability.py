import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_shap_summary(model, X_test, model_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    try:
        explainer = shap.TreeExplainer(model) #create the model to explain for models that works with trees
        shap_values = explainer.shap_values(X_test) #here is were it goes for patients and calculate the probability of infection of each variable.
        
        # --- CORRECCIÓ PER AL RANDOM FOREST ---
        if isinstance(shap_values, list):
            # Per a classificació binària en versions antigues, agafem la classe positiva
            shap_values = shap_values[1] 
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Si és un array 3D (mostres, variables, classes), s'ha d'agafar la classe positiva [:, :, 1]
            shap_values = shap_values[:, :, 1]
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=20, show=False) #generate graph with a maximum of 20 variables
        plt.savefig(os.path.join(output_path, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        plt.close()
        print(f"Plot generated {model_name}!!!!")
    except Exception as e:
        print(f"SHAP no work {model_name} //// TreeExplainer. Error: {e}")


def generate_shap_dnn(model, X_train, X_test, model_name, output_path):
    os.makedirs(output_path, exist_ok=True)

    background = X_train.values[:100] #Select background data for the DeepExplainer, we take the first 100 samples of the training set to be the background data for the SHAP values calculation
    
    try:
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_test.values)
        
        if isinstance(shap_values, list): #For in case is a 4D array, we take the first one
            shap_v = shap_values[0]
        else:
            shap_v = shap_values

        # --- CORRECCIÓ PER A LA DNN ---
        # Si l'array té forma (mostres, variables, 1), eliminem la tercera dimensió residual (squeeze)
        if isinstance(shap_v, np.ndarray) and len(shap_v.shape) == 3 and shap_v.shape[-1] == 1:
            shap_v = np.squeeze(shap_v, axis=-1)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_v, X_test, plot_type="dot", max_display=20, show=False)
        plt.savefig(os.path.join(output_path, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        plt.close()
        print(f"Plot generated {model_name}!!!!!!")
        
    except Exception as e:
        print(f"SHAP no work {model_name} //// DeepExplainer. Error: {e}")