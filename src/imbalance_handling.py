"""
imbalance_handling.py - Handling Imbalanced Labels in Graph Classification

This module implements multiple strategies for handling class imbalance:
1. Class weighting (for both Classic ML and GNNs)
2. Oversampling techniques (SMOTE for features, graph duplication)
3. Undersampling techniques
4. Hybrid approaches (SMOTEENN, SMOTETomek)
5. Focal Loss for GNNs
6. Evaluation with imbalance-aware metrics
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import copy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef,
    precision_recall_curve, average_precision_score, confusion_matrix,
    classification_report
)

# Try to import imbalanced-learn (optional dependency)
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Some resampling methods unavailable.")
    print("Install with: pip install imbalanced-learn")


# =============================================================================
# Class Imbalance Analysis
# =============================================================================

def analyze_class_distribution(labels, dataset_name="Dataset"):
    """
    Analyze and report class distribution.
    
    Args:
        labels: Array or list of labels
        dataset_name: Name for display
        
    Returns:
        Dictionary with distribution statistics
    """
    labels = np.array(labels)
    class_counts = Counter(labels)
    total = len(labels)
    
    print(f"\n{'='*50}")
    print(f"Class Distribution Analysis: {dataset_name}")
    print(f"{'='*50}")
    
    stats = {
        'total_samples': total,
        'n_classes': len(class_counts),
        'class_counts': dict(class_counts),
        'class_ratios': {},
        'imbalance_ratio': 0
    }
    
    majority_count = max(class_counts.values())
    minority_count = min(class_counts.values())
    stats['imbalance_ratio'] = majority_count / minority_count
    
    print(f"Total samples: {total}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"\nClass distribution:")
    
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        ratio = count / total
        stats['class_ratios'][cls] = ratio
        bar = '█' * int(ratio * 40)
        print(f"  Class {cls}: {count:4d} ({ratio*100:5.1f}%) {bar}")
    
    print(f"\nImbalance ratio (majority/minority): {stats['imbalance_ratio']:.2f}")
    
    if stats['imbalance_ratio'] > 1.5:
        print("⚠️  Dataset shows class imbalance!")
        if stats['imbalance_ratio'] > 3:
            print("⚠️  Severe imbalance detected - consider using resampling or weighted loss")
    else:
        print("✓ Dataset is relatively balanced")
    
    return stats


def compute_class_weights_from_labels(labels):
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        labels: Array of labels
        
    Returns:
        Dictionary mapping class to weight
    """
    labels = np.array(labels)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
    
    print("\nComputed class weights:")
    for cls, weight in weight_dict.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    return weight_dict


# =============================================================================
# Resampling for Feature-Based Methods (Classic ML)
# =============================================================================

def apply_smote(X_train, y_train, random_state=42, sampling_strategy='auto'):
    """
    Apply SMOTE oversampling to feature vectors.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        sampling_strategy: Sampling strategy ('auto', 'minority', or dict)
        
    Returns:
        X_resampled, y_resampled
    """
    if not IMBLEARN_AVAILABLE:
        print("SMOTE not available. Returning original data.")
        return X_train, y_train
    
    print("\nApplying SMOTE oversampling...")
    print(f"  Before: {Counter(y_train)}")
    
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"  After:  {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def apply_adasyn(X_train, y_train, random_state=42):
    """
    Apply ADASYN (Adaptive Synthetic Sampling).
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        X_resampled, y_resampled
    """
    if not IMBLEARN_AVAILABLE:
        print("ADASYN not available. Returning original data.")
        return X_train, y_train
    
    print("\nApplying ADASYN oversampling...")
    print(f"  Before: {Counter(y_train)}")
    
    adasyn = ADASYN(random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    
    print(f"  After:  {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def apply_random_undersampling(X_train, y_train, random_state=42, sampling_strategy='auto'):
    """
    Apply random undersampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        sampling_strategy: Sampling strategy
        
    Returns:
        X_resampled, y_resampled
    """
    if not IMBLEARN_AVAILABLE:
        print("RandomUnderSampler not available. Returning original data.")
        return X_train, y_train
    
    print("\nApplying Random Undersampling...")
    print(f"  Before: {Counter(y_train)}")
    
    rus = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"  After:  {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def apply_smoteenn(X_train, y_train, random_state=42):
    """
    Apply SMOTEENN (SMOTE + Edited Nearest Neighbors).
    Combines oversampling with cleaning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        X_resampled, y_resampled
    """
    if not IMBLEARN_AVAILABLE:
        print("SMOTEENN not available. Returning original data.")
        return X_train, y_train
    
    print("\nApplying SMOTEENN (SMOTE + ENN cleaning)...")
    print(f"  Before: {Counter(y_train)}")
    
    smoteenn = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)
    
    print(f"  After:  {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


def apply_smotetomek(X_train, y_train, random_state=42):
    """
    Apply SMOTETomek (SMOTE + Tomek Links removal).
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        X_resampled, y_resampled
    """
    if not IMBLEARN_AVAILABLE:
        print("SMOTETomek not available. Returning original data.")
        return X_train, y_train
    
    print("\nApplying SMOTETomek...")
    print(f"  Before: {Counter(y_train)}")
    
    smotetomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smotetomek.fit_resample(X_train, y_train)
    
    print(f"  After:  {Counter(y_resampled)}")
    
    return X_resampled, y_resampled


# =============================================================================
# Graph-Level Resampling (for GNNs)
# =============================================================================

def oversample_graphs(dataset, indices, labels, random_state=42):
    """
    Oversample minority class graphs by duplication.
    
    Args:
        dataset: PyTorch Geometric dataset
        indices: Indices of graphs to use
        labels: Labels for the graphs
        random_state: Random seed
        
    Returns:
        List of graph Data objects (oversampled)
    """
    np.random.seed(random_state)
    
    labels = np.array(labels)
    class_counts = Counter(labels)
    majority_count = max(class_counts.values())
    
    print("\nOversampling graphs (duplication)...")
    print(f"  Before: {class_counts}")
    
    oversampled_graphs = []
    
    for idx, label in zip(indices, labels):
        oversampled_graphs.append(dataset[idx])
    
    # Oversample minority classes
    for cls in class_counts:
        if class_counts[cls] < majority_count:
            # Find indices of this class
            cls_indices = [i for i, l in enumerate(labels) if l == cls]
            n_to_add = majority_count - class_counts[cls]
            
            # Randomly select and duplicate
            selected = np.random.choice(cls_indices, size=n_to_add, replace=True)
            for idx in selected:
                # Deep copy the graph
                original = dataset[indices[idx]]
                duplicate = Data(
                    x=original.x.clone(),
                    edge_index=original.edge_index.clone(),
                    y=original.y.clone()
                )
                if hasattr(original, 'edge_attr') and original.edge_attr is not None:
                    duplicate.edge_attr = original.edge_attr.clone()
                oversampled_graphs.append(duplicate)
    
    new_labels = [g.y.item() for g in oversampled_graphs]
    print(f"  After:  {Counter(new_labels)}")
    
    return oversampled_graphs


def undersample_graphs(dataset, indices, labels, random_state=42):
    """
    Undersample majority class graphs.
    
    Args:
        dataset: PyTorch Geometric dataset
        indices: Indices of graphs to use
        labels: Labels for the graphs
        random_state: Random seed
        
    Returns:
        List of graph indices (undersampled)
    """
    np.random.seed(random_state)
    
    labels = np.array(labels)
    class_counts = Counter(labels)
    minority_count = min(class_counts.values())
    
    print("\nUndersampling graphs...")
    print(f"  Before: {class_counts}")
    
    undersampled_indices = []
    
    for cls in class_counts:
        cls_indices = [indices[i] for i, l in enumerate(labels) if l == cls]
        
        if class_counts[cls] > minority_count:
            # Randomly select minority_count samples
            selected = np.random.choice(cls_indices, size=minority_count, replace=False)
            undersampled_indices.extend(selected)
        else:
            undersampled_indices.extend(cls_indices)
    
    # Get new labels
    new_labels = [dataset[i].y.item() for i in undersampled_indices]
    print(f"  After:  {Counter(new_labels)}")
    
    return undersampled_indices


# =============================================================================
# Weighted Loss Functions for GNNs
# =============================================================================

class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with class weights."""
    
    def __init__(self, class_weights):
        """
        Args:
            class_weights: Dictionary or tensor of class weights
        """
        super().__init__()
        if isinstance(class_weights, dict):
            weights = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())])
        else:
            weights = torch.tensor(class_weights)
        self.register_buffer('weight', weights.float())
        
    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (list, tensor, or None for equal weights)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha).float()
            elif isinstance(alpha, dict):
                self.alpha = torch.tensor([alpha[i] for i in sorted(alpha.keys())]).float()
            else:
                self.alpha = alpha
        else:
            self.alpha = None
            
    def forward(self, input, target):
        """
        Args:
            input: Model output logits (N, C)
            target: Ground truth labels (N,)
        """
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = p if correct class
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != input.device:
                self.alpha = self.alpha.to(input.device)
            alpha_t = self.alpha[target]
            focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (2019)
    """
    
    def __init__(self, samples_per_class, beta=0.9999, loss_type='focal', gamma=2.0):
        """
        Args:
            samples_per_class: List of sample counts per class
            beta: Hyperparameter for effective number calculation
            loss_type: 'focal' or 'softmax'
            gamma: Gamma for focal loss
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        
        # Compute effective number and weights
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)  # Normalize
        
        self.register_buffer('weights', torch.tensor(weights).float())
        
    def forward(self, input, target):
        weights_for_samples = self.weights[target]
        
        if self.loss_type == 'focal':
            ce_loss = F.cross_entropy(input, target, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma
            loss = weights_for_samples * focal_weight * ce_loss
        else:
            loss = weights_for_samples * F.cross_entropy(input, target, reduction='none')
        
        return loss.mean()


# =============================================================================
# Imbalance-Aware Evaluation Metrics
# =============================================================================

def evaluate_with_imbalance_metrics(y_true, y_pred, y_prob=None):
    """
    Comprehensive evaluation with imbalance-aware metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        # Standard metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred),  # Matthews Correlation Coefficient
    }
    
    # Per-class metrics
    for cls in np.unique(y_true):
        y_true_binary = (np.array(y_true) == cls).astype(int)
        y_pred_binary = (np.array(y_pred) == cls).astype(int)
        
        metrics[f'precision_class_{cls}'] = precision_score(y_true_binary, y_pred_binary)
        metrics[f'recall_class_{cls}'] = recall_score(y_true_binary, y_pred_binary)
        metrics[f'f1_class_{cls}'] = f1_score(y_true_binary, y_pred_binary)
    
    # AUC metrics (if probabilities available)
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)  # PR-AUC
        except:
            metrics['auc_roc'] = np.nan
            metrics['auc_pr'] = np.nan
    
    return metrics


def print_classification_report_extended(y_true, y_pred, y_prob=None):
    """
    Print extended classification report with imbalance metrics.
    """
    print("\n" + "=" * 60)
    print("EXTENDED CLASSIFICATION REPORT")
    print("=" * 60)
    
    # Standard sklearn report
    print("\nPer-Class Metrics:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Additional metrics
    metrics = evaluate_with_imbalance_metrics(y_true, y_pred, y_prob)
    
    print("\nImbalance-Aware Metrics:")
    print(f"  Balanced Accuracy:     {metrics['balanced_accuracy']:.4f}")
    print(f"  Matthews Corr. Coef.:  {metrics['mcc']:.4f}")
    print(f"  Macro F1:              {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:           {metrics['f1_weighted']:.4f}")
    
    if 'auc_roc' in metrics and not np.isnan(metrics['auc_roc']):
        print(f"  AUC-ROC:               {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:                {metrics['auc_pr']:.4f}")
    
    return metrics


# =============================================================================
# Threshold Optimization for Imbalanced Data
# =============================================================================

def optimize_threshold(y_true, y_prob, metric='f1'):
    """
    Find optimal classification threshold for imbalanced data.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'mcc')
        
    Returns:
        Optimal threshold and best metric value
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (np.array(y_prob) >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, average='macro')
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'mcc':
            score = matthews_corrcoef(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, average='macro')
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    print(f"\nOptimal threshold ({metric}): {best_threshold:.2f}")
    print(f"Best {metric} score: {best_score:.4f}")
    
    return best_threshold, best_score


# =============================================================================
# Experiment Runner with Imbalance Handling
# =============================================================================

def run_imbalance_experiments(dataset, save_dir='./results/imbalance'):
    """
    Run experiments comparing different imbalance handling strategies.
    
    Args:
        dataset: PyTorch Geometric dataset
        save_dir: Directory to save results
    """
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("IMBALANCED DATA HANDLING EXPERIMENTS")
    print("=" * 70)
    
    # Analyze class distribution
    labels = [data.y.item() for data in dataset]
    stats = analyze_class_distribution(labels, "MUTAG Dataset")
    
    # Compute class weights
    class_weights = compute_class_weights_from_labels(labels)
    
    # Split data
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    
    print(f"\nTrain distribution: {Counter(y_train)}")
    print(f"Test distribution:  {Counter(y_test)}")
    
    # Import and use gSpan features
    from q1_gspan_classic_ml import GSpanMiner, construct_feature_vectors
    from data_utils import pyg_to_networkx_graphs
    
    graphs, all_labels = pyg_to_networkx_graphs(dataset)
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    
    # Mine patterns
    miner = GSpanMiner(min_support=0.1)
    patterns = miner.mine(train_graphs)
    
    X_train = construct_feature_vectors(train_graphs, patterns)
    X_test = construct_feature_vectors(test_graphs, patterns)
    
    results = []
    
    # =====================================================
    # Experiment 1: Baseline (No handling)
    # =====================================================
    print("\n" + "=" * 50)
    print("Experiment 1: Baseline (No Imbalance Handling)")
    print("=" * 50)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_with_imbalance_metrics(y_test, y_pred, y_prob)
    metrics['method'] = 'Baseline'
    results.append(metrics)
    
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    # =====================================================
    # Experiment 2: Class Weights
    # =====================================================
    print("\n" + "=" * 50)
    print("Experiment 2: Class Weighting")
    print("=" * 50)
    
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_with_imbalance_metrics(y_test, y_pred, y_prob)
    metrics['method'] = 'Class Weights'
    results.append(metrics)
    
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    # =====================================================
    # Experiment 3: SMOTE
    # =====================================================
    if IMBLEARN_AVAILABLE:
        print("\n" + "=" * 50)
        print("Experiment 3: SMOTE Oversampling")
        print("=" * 50)
        
        X_train_smote, y_train_smote = apply_smote(X_train, y_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_smote, y_train_smote)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_with_imbalance_metrics(y_test, y_pred, y_prob)
        metrics['method'] = 'SMOTE'
        results.append(metrics)
        
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    # =====================================================
    # Experiment 4: Random Undersampling
    # =====================================================
    if IMBLEARN_AVAILABLE:
        print("\n" + "=" * 50)
        print("Experiment 4: Random Undersampling")
        print("=" * 50)
        
        X_train_under, y_train_under = apply_random_undersampling(X_train, y_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_under, y_train_under)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_with_imbalance_metrics(y_test, y_pred, y_prob)
        metrics['method'] = 'Undersampling'
        results.append(metrics)
        
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    # =====================================================
    # Experiment 5: SMOTEENN (Hybrid)
    # =====================================================
    if IMBLEARN_AVAILABLE:
        print("\n" + "=" * 50)
        print("Experiment 5: SMOTEENN (Hybrid)")
        print("=" * 50)
        
        X_train_smoteenn, y_train_smoteenn = apply_smoteenn(X_train, y_train)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_smoteenn, y_train_smoteenn)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_with_imbalance_metrics(y_test, y_pred, y_prob)
        metrics['method'] = 'SMOTEENN'
        results.append(metrics)
        
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    # =====================================================
    # Experiment 6: Threshold Optimization
    # =====================================================
    print("\n" + "=" * 50)
    print("Experiment 6: Threshold Optimization")
    print("=" * 50)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    best_thresh, _ = optimize_threshold(y_test, y_prob, metric='f1')
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    
    metrics = evaluate_with_imbalance_metrics(y_test, y_pred_opt, y_prob)
    metrics['method'] = 'Threshold Optimization'
    results.append(metrics)
    
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    
    # =====================================================
    # Summary
    # =====================================================
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("IMBALANCE HANDLING COMPARISON SUMMARY")
    print("=" * 70)
    
    summary_cols = ['method', 'balanced_accuracy', 'f1_macro', 'mcc', 'auc_roc']
    available_cols = [c for c in summary_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))
    
    # Save results
    results_df.to_csv(f'{save_dir}/imbalance_comparison.csv', index=False)
    
    # Find best method
    best_method = results_df.loc[results_df['balanced_accuracy'].idxmax(), 'method']
    print(f"\nBest method (by balanced accuracy): {best_method}")
    
    return results_df


if __name__ == "__main__":
    from data_utils import load_mutag_dataset
    
    dataset, info = load_mutag_dataset()
    results = run_imbalance_experiments(dataset)
