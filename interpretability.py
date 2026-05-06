import shap
import matplotlib.pyplot as plt
import os

def generate_shap_summary(model, X_test, model_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    try:
        explainer = shap.TreeExplainer(model) #create the model to explain for models that works with trees
        shap_values = explainer.shap_values(X_test) #here is were it goes for patients and calculate the probability of infection of each variable.
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # For binary classification, we take the SHAP values for the positive classfor random forest
            
        plt.figure()
        shap.summary_plot(shap_values, X_test, max_display=20, show=False) #generate graph with a maximum of 20 variables
        plt.savefig(os.path.join(output_path, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        plt.close()
        print(f"Plot generated {model_name}!!!!")
    except Exception as e:
        print(f"SHAP no work {model_name} //// TreeExplainer. Error: {e}")