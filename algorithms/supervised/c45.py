import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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


def entropy(y):
    """Calculate entropy of labels"""
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


def information_gain(X, y, feature_idx, threshold):
    """
    Calculate Information Gain for a split
    Information Gain = Parent Entropy - Weighted Child Entropy
    """
    parent_entropy = entropy(y)
    
    # Split data
    left_mask = X.iloc[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0
    
    # Calculate weighted entropy
    n = len(y)
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    
    child_entropy = (np.sum(left_mask) / n * left_entropy + 
                     np.sum(right_mask) / n * right_entropy)
    
    return parent_entropy - child_entropy


def split_information(X, feature_idx, threshold):
    """
    Calculate Split Information for Gain Ratio
    Split Info = -∑ (|Sj|/|S|) * log2(|Sj|/|S|)
    Where Sj are the subsets after splitting
    """
    n = len(X)
    left_mask = X.iloc[:, feature_idx] <= threshold
    right_mask = ~left_mask
    
    left_count = np.sum(left_mask)
    right_count = np.sum(right_mask)
    
    if left_count == 0 or right_count == 0:
        return 1e-9  # Avoid division by zero
    
    # Calculate split information
    left_prob = left_count / n
    right_prob = right_count / n
    
    split_info = -(left_prob * np.log2(left_prob + 1e-9) + 
                   right_prob * np.log2(right_prob + 1e-9))
    
    return split_info if split_info > 0 else 1e-9


def gain_ratio(X, y, feature_idx, threshold):
    """
    Calculate Gain Ratio (C4.5 criterion)
    Gain Ratio = Information Gain / Split Information
    
    This is the key difference from ID3:
    - ID3 uses Information Gain only
    - C4.5 uses Gain Ratio to avoid bias toward attributes with many values
    """
    ig = information_gain(X, y, feature_idx, threshold)
    si = split_information(X, feature_idx, threshold)
    
    return ig / si if si > 0 else 0


class DecisionNode:
    """Node in the decision tree"""
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.value = value  # Class value if leaf node


def build_tree(X, y, depth=0, max_depth=10):
    """Build decision tree recursively"""
    n_samples, n_features = X.shape
    
    # Convert y to numpy array and ensure int type
    y_array = np.array(y).astype(int)
    n_classes = len(np.unique(y_array))
    
    # Stopping criteria
    if depth >= max_depth or n_classes == 1 or n_samples < 2:
        # Leaf node
        leaf_value = np.bincount(y_array).argmax()
        return DecisionNode(value=int(leaf_value))
    
    # Find best split using Gain Ratio (C4.5 criterion)
    # At each node: compute Gain Ratio for each attribute
    # Select attribute with maximum Gain Ratio
    best_gain_ratio = -1
    best_feature = None
    best_threshold = None
    
    for feature_idx in range(n_features):
        feature_values = X.iloc[:, feature_idx]
        thresholds = np.unique(feature_values)
        
        for threshold in thresholds:
            # Use Gain Ratio instead of Information Gain (C4.5 improvement)
            gr = gain_ratio(X, y_array, feature_idx, threshold)
            if gr > best_gain_ratio:
                best_gain_ratio = gr
                best_feature = feature_idx
                best_threshold = threshold
    
    # If no good split found, create leaf
    if best_gain_ratio == -1 or best_feature is None:
        leaf_value = np.bincount(y_array).argmax()
        return DecisionNode(value=int(leaf_value))
    
    # Split data
    left_mask = X.iloc[:, best_feature] <= best_threshold
    right_mask = ~left_mask
    
    # Build subtrees
    left_subtree = build_tree(X[left_mask].reset_index(drop=True), 
                               y_array[left_mask], 
                               depth + 1, max_depth)
    right_subtree = build_tree(X[right_mask].reset_index(drop=True), 
                                y_array[right_mask], 
                                depth + 1, max_depth)
    
    return DecisionNode(best_feature, best_threshold, left_subtree, right_subtree)


def predict_tree(node, x):
    """Predict class for a single sample"""
    if node.value is not None:
        return node.value
    
    if x[node.feature_idx] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)


@st.cache_data
def run_c45(X_train, X_test, y_train, y_test, params):
    """
    C4.5 Decision Tree Classification
    
    Training Phase:
    1. At each node, compute Gain Ratio for each attribute:
       - Total Entropy
       - Information Gain per attribute
       - Split Information
       - Gain Ratio = Information Gain / Split Information
    2. Select attribute with maximum Gain Ratio as root/split node
    3. Split dataset based on attribute values
    4. For each subset:
       - If pure (all instances one class): create leaf
       - Otherwise: repeat recursively
    
    Classification Phase:
    1. Start from root
    2. At each node: follow branch matching instance's attribute value
    3. Continue until reaching a leaf
    4. Leaf indicates predicted class
    """
    max_depth = params.get('max_depth', 10)
    
    # Ensure y_train is integer type
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    
    # Build tree
    tree = build_tree(X_train, y_train_int, max_depth=max_depth)
    
    # Make predictions
    predictions = []
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test.iloc[i].values)
        predictions.append(int(pred))
    
    y_pred = np.array(predictions, dtype=int)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_int.values, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_int, y_pred)
    classes = np.unique(y_test_int)
    cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
    
    # Create confusion matrix heatmap
    figures = {}
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {c}" for c in classes],
        y=[f"Class {c}" for c in classes],
        title="Confusion Matrix - C4.5 Decision Tree",
        color_continuous_scale='Purples',
        text_auto=True
    )
    fig_cm.update_layout(height=400)
    figures['Confusion Matrix Heatmap'] = fig_cm
    
    return y_pred, metrics, cm_df, figures
