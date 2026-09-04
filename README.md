**PREDICTION OF SURGICAL SITE INFECTIONS (SSI) IN TRAUMATOLOGY**

This project is the practical component of the Final Master Project (FMP). Its objective is to compare various Machine Learning and Deep Learning models for the retrospective detection of SSI in trauma surgery.

**PROJECT STRUCTURE**

├── config.json               Centralized configuration (paths, hyperparameters)

├── preprocessing.py          Data cleaning, normalization, and balancing (SMOTE)

├── descriptive_analysis.py   Perform a descriptive analysis of the variables

├── requirements.txt          Control version of the packages

├── boostrap_experimental.py  Main script to train and test all the models

├── plots_results.py          Script to generate the plots for the article

├── old_scripts/              Old scripts used during the FMP project

├── models/                   Model definition package

│   ├── **init**.py

│   └── dl_models.py          Dense Neural Network (DNN) with TensorFlow/Keras

├── utils/                    Statistical utilities

│   ├── **init**.py

│   └── metrics.py            Metrics calculation and threshold optimization

├── results/                  Output: SHAP plots and CSV reports (automatically generated)

└── artifacts/                Output: Trained models (.pkl / .keras) (automatically generated)


**INSTALLATION AND SETUP**

1. Clone repository
   git clone https://github.com/MarcTomasMoncu/FMP_code.git
   cd FMP_code

2. Create virtual environment and install dependencies
   python3 -m venv venv
   source venv/bin/activate 
   pip install -r requirements.txt

**CONFIGURATION**

Variables are controlled through config.json, such as sensitivity, activation of SMOTENC for synthetic data, and whether certain model columns should be ignored.

**USAGE**

From the root folder of the repository, simply run in the terminal:
  $python3 boostrap_experimental.py

**RESULTS AND INTERPRETATION**

In the results folder, the following are automatically generated:

1. taula_resultats_boostrap.csv: Comparison of all models.


**EXTRA ANALYSIS**

If a descriptive analysis is required, you should run the descriptive_analysis.py script from your terminal using the following command:

   $python3 descriptive_analysis.py

Once executed, two PNG images will be generated and saved directly into the results/ folder:

1. general descriptive summary of the dataset.

2. comparative summary stratifying the data by infection vs. non-infection.

