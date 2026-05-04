import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from utils.metrics import calculate_metrics

def build_dnn_model(input_dim, lr=1e-4, dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  metrics=['accuracy'])
    return model

def cross_validate_dnn(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics_list = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        y_val_pred = model.predict(X_val).ravel()
        metrics = calculate_metrics(y_val, y_val_pred)
        metrics_list.append(metrics)
    return metrics_list

def train_and_evaluate_dnn(model, X_train, y_train, X_test, y_test, model_file):
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    model.save(model_file)
    y_test_pred = model.predict(X_test).ravel()
    metrics = calculate_metrics(y_test, y_test_pred)
    return metrics, y_test_pred