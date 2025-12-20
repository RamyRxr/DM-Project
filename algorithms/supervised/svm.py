import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn import svm


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics: Precision, Recall, F-Measure, and Accuracy
    
    For each class:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F-Measure = 2 * (Precision * Recall) / (Precision + Recall)
    
    Overall:
    - Accuracy = Correct Predictions / Total Predictions
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Initialize metrics
    precision_per_class = {}
    recall_per_class = {}
    f_measure_per_class = {}
    
    # Calculate for each class
    for cls in classes:
        # True Positives: Predicted as cls AND actually cls
        tp = np.sum((y_true == cls) & (y_pred == cls))
        
        # False Positives: Predicted as cls BUT actually NOT cls
        fp = np.sum((y_true != cls) & (y_pred == cls))
        
        # False Negatives: Predicted as NOT cls BUT actually cls
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
    
    # Calculate macro averages (average across all classes)
    avg_precision = np.mean(list(precision_per_class.values()))
    avg_recall = np.mean(list(recall_per_class.values()))
    avg_f_measure = np.mean(list(f_measure_per_class.values()))
    
    # Accuracy: Total correct predictions / Total predictions
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        'Accuracy': accuracy,
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F-Measure': avg_f_measure
    }


@st.cache_data
def run_svm(X_train, X_test, y_train, y_test, params):
    """
    Support Vector Machine (SVM) Classification
    
    SVM finds the optimal hyperplane that maximizes the margin between classes.
    
    Key Concepts:
    - Support Vectors: Data points closest to the decision boundary
    - Hyperplane: Decision boundary separating classes
    - Margin: Distance from hyperplane to nearest support vectors
    - Kernel: Function to transform data into higher dimensions (linear, RBF, poly)
    
    Training Phase:
    1. Transform data using kernel function (if non-linear)
    2. Find optimal hyperplane that maximizes margin
    3. Identify support vectors (critical training points)
    
    Prediction Phase:
    1. For each test point, determine which side of hyperplane it falls on
    2. Assign class based on hyperplane decision
    
    Parameters:
    - kernel: Type of kernel function ('linear', 'rbf', 'poly')
    - C: Regularization parameter (controls margin vs misclassification trade-off)
    """
    # Get parameters
    kernel = params.get('kernel', 'rbf')
    C = params.get('C', 1.0)
    
    # Ensure integer types for classification
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    
    # Create SVM classifier
    # C: Regularization - smaller C = wider margin, more misclassifications allowed
    # kernel: 'linear' for linearly separable, 'rbf' for non-linear
    clf = svm.SVC(kernel=kernel, C=C, random_state=42)
    
    # Train the model
    clf.fit(X_train, y_train_int)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred = y_pred.astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_int.values, y_pred)
    
    # Add SVM-specific information
    metrics['Support Vectors'] = clf.n_support_.sum()
    metrics['Kernel'] = kernel
    metrics['C Parameter'] = C
    
    # Confusion matrix
    cm = confusion_matrix(y_test_int, y_pred)
    classes_list = np.unique(y_test_int)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes_list], columns=[f"Pred {c}" for c in classes_list])
    
    # Create visualizations
    figures = {}
    
    # Confusion matrix heatmap
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {c}" for c in classes_list],
        y=[f"Class {c}" for c in classes_list],
        title=f"Confusion Matrix - SVM (kernel={kernel}, C={C})",
        color_continuous_scale='Blues',
        text_auto=True
    )
    fig_cm.update_layout(height=400)
    figures['Confusion Matrix Heatmap'] = fig_cm
    
    # Create support vector information visualization
    support_vector_counts = clf.n_support_
    fig_sv = px.bar(
        x=[f"Class {c}" for c in classes_list],
        y=support_vector_counts,
        labels={'x': 'Class', 'y': 'Number of Support Vectors'},
        title=f"Support Vectors per Class (Total: {clf.n_support_.sum()})",
        color=support_vector_counts,
        color_continuous_scale='Viridis'
    )
    fig_sv.update_layout(height=350)
    figures['Support Vectors Distribution'] = fig_sv
    
    return y_pred, metrics, cm_df, figures
