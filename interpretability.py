import shap
import matplotlib.pyplot as plt
import os

def generate_shap_summary(model, X_test, model_name, output_path):
    """Genera i guarda l'explicació SHAP del model."""
    os.makedirs(output_path, exist_ok=True)
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        plt.figure()
        shap.summary_plot(shap_values, X_test, max_display=20, show=False) 
        plt.savefig(os.path.join(output_path, f"{model_name}_shap_summary.png"), bbox_inches='tight')
        plt.close()
        print(f"Grafic SHAP generat per a {model_name}.")
    except Exception as e:
        print(f"SHAP no suportat directament per {model_name} amb TreeExplainer. Error: {e}")