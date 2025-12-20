import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix


def euclidean_distance(x1, x2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((x1 - x2) ** 2))


def knn_predict(X_train, y_train, x_test, k):
    """
    Predict class for a single test instance using k-NN
    """
    # Calculate distances to all training samples
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train.iloc[i].values, x_test.values)
        distances.append((dist, y_train.iloc[i]))
    
    # Sort by distance and get k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Get labels of k nearest neighbors
    k_nearest_labels = [label for _, label in k_nearest]
    
    # Return most common label (majority vote)
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    return unique_labels[np.argmax(counts)]


def calculate_metrics(y_true, y_pred):
    """
    Calculate precision, recall, F-measure, and accuracy
    """
    # Get unique classes
    classes = np.unique(y_true)
    
    # Initialize metrics
    precision_per_class = {}
    recall_per_class = {}
    f_measure_per_class = {}
    
    # Calculate for each class
    for cls in classes:
        # True Positives, False Positives, False Negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class[cls] = precision
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class[cls] = recall
        
        # F-Measure = 2 * (Precision * Recall) / (Precision + Recall)
        f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f_measure_per_class[cls] = f_measure
    
    # Calculate averages (macro average)
    avg_precision = np.mean(list(precision_per_class.values()))
    avg_recall = np.mean(list(recall_per_class.values()))
    avg_f_measure = np.mean(list(f_measure_per_class.values()))
    
    # Accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'Accuracy': accuracy,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F-Measure': avg_f_measure
    }


@st.cache_data
def run_knn(X_train, X_test, y_train, y_test, params):
    """
    Run k-NN classification
    """
    k = params.get('k', 5)
    
    # Ensure integer types for classification
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    
    # Make predictions
    predictions = []
    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train_int, X_test.iloc[i], k)
        predictions.append(int(pred))
    
    y_pred = np.array(predictions, dtype=int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_int.values, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_int, y_pred)
    classes = np.unique(y_test_int)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
    
    # Test multiple k values for comparison
    k_values = list(range(1, min(21, len(X_train))))
    accuracies = []
    precisions = []
    recalls = []
    f_measures = []
    
    for k_val in k_values:
        preds = []
        for i in range(len(X_test)):
            pred = knn_predict(X_train, y_train_int, X_test.iloc[i], k_val)
            preds.append(int(pred))
        y_p = np.array(preds, dtype=int)
        m = calculate_metrics(y_test_int.values, y_p)
        accuracies.append(m['Accuracy'])
        precisions.append(m['Precision'])
        recalls.append(m['Recall'])
        f_measures.append(m['F-Measure'])
    
    # Create figures
    figures = {}
    
    # Accuracy vs k
    fig_accuracy = px.line(
        x=k_values,
        y=accuracies,
        markers=True,
        title=f'Accuracy vs k (Current k={k})',
        labels={'x': 'k (number of neighbors)', 'y': 'Accuracy'}
    )
    fig_accuracy.add_vline(x=k, line_dash="dash", line_color="red", annotation_text=f"Current k={k}")
    fig_accuracy.update_layout(height=400)
    figures['Accuracy vs k'] = fig_accuracy
    
    # Precision vs k
    fig_precision = px.line(
        x=k_values,
        y=precisions,
        markers=True,
        title='Precision vs k',
        labels={'x': 'k (number of neighbors)', 'y': 'Precision'}
    )
    fig_precision.add_vline(x=k, line_dash="dash", line_color="red", annotation_text=f"k={k}")
    fig_precision.update_layout(height=400)
    figures['Precision vs k'] = fig_precision
    
    # Confusion matrix heatmap
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {c}" for c in classes],
        y=[f"Class {c}" for c in classes],
        title="Confusion Matrix",
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig_cm.update_layout(height=400)
    figures['Confusion Matrix Heatmap'] = fig_cm
    
    return y_pred, metrics, cm_df, figures
