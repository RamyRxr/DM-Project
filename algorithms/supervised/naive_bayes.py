import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix


def calculate_metrics(y_true, y_pred):
    """Calculate precision, recall, F-measure, and accuracy"""
    classes = np.unique(y_true)
    
    precision_per_class = {}
    recall_per_class = {}
    f_measure_per_class = {}
    
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class[cls] = precision
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class[cls] = recall
        
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f_measure_per_class[cls] = f_measure
    
    avg_precision = np.mean(list(precision_per_class.values()))
    avg_recall = np.mean(list(recall_per_class.values()))
    avg_f_measure = np.mean(list(f_measure_per_class.values()))
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'Accuracy': accuracy,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F-Measure': avg_f_measure
    }


@st.cache_data
def run_naive_bayes(X_train, X_test, y_train, y_test, params):
    """
    Naive Bayes classification (Gaussian Naive Bayes)
    P(class|features) = P(features|class) * P(class) / P(features)
    """
    classes = np.unique(y_train)
    n_features = X_train.shape[1]
    
    # Calculate prior probabilities P(class)
    priors = {}
    for cls in classes:
        priors[cls] = np.sum(y_train == cls) / len(y_train)
    
    # Calculate mean and std for each feature per class
    means = {}
    stds = {}
    for cls in classes:
        X_cls = X_train[y_train == cls]
        means[cls] = X_cls.mean(axis=0).values
        stds[cls] = X_cls.std(axis=0).values + 1e-6  # Add small value to avoid division by zero
    
    # Predict function for Naive Bayes
    def predict_naive_bayes(x):
        posteriors = {}
        
        for cls in classes:
            # Start with prior
            posterior = np.log(priors[cls])
            
            # Add log likelihood for each feature (Gaussian)
            for i in range(n_features):
                mean = means[cls][i]
                std = stds[cls][i]
                # Gaussian probability density
                likelihood = -0.5 * np.log(2 * np.pi * std**2) - ((x[i] - mean)**2) / (2 * std**2)
                posterior += likelihood
            
            posteriors[cls] = posterior
        
        # Return class with highest posterior
        return max(posteriors, key=posteriors.get)
    
    # Make predictions
    predictions = []
    for i in range(len(X_test)):
        pred = predict_naive_bayes(X_test.iloc[i].values)
        predictions.append(pred)
    
    y_pred = np.array(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test.values, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    classes_list = np.unique(y_test)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes_list], columns=[f"Pred {c}" for c in classes_list])
    
    # Create confusion matrix heatmap
    figures = {}
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {c}" for c in classes_list],
        y=[f"Class {c}" for c in classes_list],
        title="Confusion Matrix - Naive Bayes",
        color_continuous_scale='Greens',
        text_auto=True
    )
    fig_cm.update_layout(height=400)
    figures['Confusion Matrix Heatmap'] = fig_cm
    
    return y_pred, metrics, cm_df, figures
