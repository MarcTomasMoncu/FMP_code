import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from utils.metrics import calculate_metrics

def build_dnn_model(input_dim, lr=1e-4, dropout_rate=0.5): #function to build a dense neural network model with the specified input dimension, learning rate and dropout rate
    model = Sequential() #create a sequential model, which is a linear stack of layers
    model.add(Dense(64, input_dim=input_dim, activation='relu')) #add a dense layer with 64 units, ReLU activation function and the specified input dimension
    model.add(Dense(64, activation='relu')) #add another dense layer with 64 units and ReLU activation function
    model.add(Dropout(dropout_rate)) #add a dropout layer with the specified dropout rate to prevent overfitting
    model.add(Dense(1, activation='sigmoid')) #add an output layer with 1 unit and sigmoid activation function for binary classification
    model.compile(loss='binary_crossentropy',  #compile the model with binary cross-entropy loss function, Adam optimizer with the specified learning rate and accuracy as a metric
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  metrics=['accuracy'])
    return model

def cross_validate_dnn(model, X, y, n_splits=5): 
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = [] 
    
    for train_index, val_index in skf.split(X, y): 
        X_train_fold = X.values[train_index] if hasattr(X, "values") else X[train_index]
        X_val_fold = X.values[val_index] if hasattr(X, "values") else X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # SMOTE controlat dins de la iteració
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_fold, y_train_fold)
        
        # CRÍTIC: Reconstruïm i compilem el model des de zero a cada fold 
        # per netejar la memòria dels pesos de la xarxa neural
        fold_model = build_dnn_model(input_dim=X.shape[1])
        
        # Entrenem el model fresc d'aquest fold
        fold_model.fit(X_train_res, y_train_res, epochs=20, batch_size=16, verbose=0)
        
        # Predicció neta sobre el fold de validació real
        y_val_pred = fold_model.predict(X_val_fold).ravel()
        metrics = calculate_metrics(y_val_fold, y_val_pred)
        metrics_list.append(metrics)
        
    return metrics_list

def train_and_evaluate_dnn(model, X_train, y_train, X_test, y_test, model_file): #function to train the dense neural network model on the training data, save the trained model to a file and evaluate it on the test data, returning the calculated metrics and predicted probabilities for the test set
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    model.save(model_file)
    y_test_pred = model.predict(X_test).ravel()
    metrics = calculate_metrics(y_test, y_test_pred)
    return metrics, y_test_pred