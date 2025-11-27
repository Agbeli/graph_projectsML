"""
q1_gspan_classic_ml.py - Q1: Frequent Subgraph Mining + Classic ML

This module implements:
1. gSpan algorithm for frequent subgraph mining
2. Feature vector construction based on subgraph counts
3. Classic ML models (Random Forest, SVM, Gradient Boosting)
4. Ablation studies on mining thresholds and model parameters
5. Imbalanced data handling strategies
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import time
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import networkx as nx
from networkx.algorithms import isomorphism

# Try to import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Note: imbalanced-learn not installed. Install with: pip install imbalanced-learn")

# Import local modules
from data_utils import (
    load_mutag_dataset, pyg_to_networkx_graphs, 
    split_dataset, Timer
)


# =============================================================================
# Cross-Validation Functions for Classic ML
# =============================================================================

def cross_validate_classic_ml(X, y, model, model_name, n_folds=5, random_state=42):
    """
    Perform k-fold cross-validation for classic ML models.
    
    Args:
        X: Feature matrix
        y: Labels
        model: Scikit-learn classifier
        model_name: Name of the model
        n_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        Dictionary with CV results
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'f1_weighted': 'f1_weighted',
        'f1_macro': 'f1_macro',
        'roc_auc': 'roc_auc',
    }
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation: {model_name} ({n_folds} folds)")
    print(f"{'='*60}")
    
    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=skf, scoring=scoring,
        return_train_score=True, n_jobs=-1
    )
    
    # Compile results
    results = {
        'model': model_name,
        'n_folds': n_folds,
    }
    
    for metric in scoring.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[f'{metric}_mean'] = test_scores.mean()
        results[f'{metric}_std'] = test_scores.std()
        results[f'{metric}_train_mean'] = train_scores.mean()
        
        print(f"  {metric:20s}: {test_scores.mean():.4f} ± {test_scores.std():.4f}")
    
    results['fit_time_mean'] = cv_results['fit_time'].mean()
    results['score_time_mean'] = cv_results['score_time'].mean()
    
    return results


def cross_validate_all_classic_ml(X, y, n_folds=5, random_state=42):
    """
    Run cross-validation for all classic ML models.
    
    Args:
        X: Feature matrix
        y: Labels
        n_folds: Number of CV folds
        random_state: Random seed
        
    Returns:
        DataFrame with all CV results
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state),
        'Random Forest (no weight)': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=random_state),
        'SVM (Linear)': SVC(kernel='linear', C=1.0, probability=True, class_weight='balanced', random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    }
    
    all_results = []
    
    for model_name, model in models.items():
        # For SVM, scale features
        if 'SVM' in model_name:
            from sklearn.pipeline import Pipeline
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
        
        results = cross_validate_classic_ml(X, y, model, model_name, n_folds, random_state)
        all_results.append(results)
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION COMPARISON (CLASSIC ML)")
    print(f"{'='*70}")
    
    display_cols = ['model', 'accuracy_mean', 'accuracy_std', 
                    'balanced_accuracy_mean', 'f1_macro_mean', 'roc_auc_mean']
    available_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))
    
    return results_df


class GSpanMiner:
    """
    Wrapper for gSpan frequent subgraph mining.
    Uses a simplified implementation for educational purposes.
    """
    
    def __init__(self, min_support=0.1, max_subgraph_size=10):
        """
        Args:
            min_support: Minimum support threshold (fraction of graphs)
            max_subgraph_size: Maximum number of nodes in subgraphs
        """
        self.min_support = min_support
        self.max_subgraph_size = max_subgraph_size
        self.frequent_subgraphs = []
        self.mining_time = 0
        
    def mine(self, graphs, labels=None):
        """
        Mine frequent subgraphs from a list of graphs.
        
        For simplicity, we mine node patterns and edge patterns
        as a proxy for full gSpan (which requires complex DFS code enumeration).
        
        Args:
            graphs: List of NetworkX graphs
            labels: Optional list of labels (not used in basic mining)
            
        Returns:
            List of frequent subgraph patterns (as NetworkX graphs)
        """
        start_time = time.time()
        
        n_graphs = len(graphs)
        min_count = int(self.min_support * n_graphs)
        
        patterns = []
        pattern_supports = []
        
        # Mine 1-node patterns (node labels)
        node_label_counts = defaultdict(int)
        for G in graphs:
            seen_labels = set()
            for node in G.nodes():
                label = G.nodes[node].get('label', 0)
                if label not in seen_labels:
                    node_label_counts[label] += 1
                    seen_labels.add(label)
        
        for label, count in node_label_counts.items():
            if count >= min_count:
                # Create single-node subgraph
                sg = nx.Graph()
                sg.add_node(0, label=label)
                patterns.append(sg)
                pattern_supports.append(count / n_graphs)
        
        # Mine 2-node patterns (edges with node labels)
        edge_pattern_counts = defaultdict(int)
        for G in graphs:
            seen_patterns = set()
            for u, v in G.edges():
                label_u = G.nodes[u].get('label', 0)
                label_v = G.nodes[v].get('label', 0)
                # Canonical form (smaller label first)
                pattern = tuple(sorted([label_u, label_v]))
                if pattern not in seen_patterns:
                    edge_pattern_counts[pattern] += 1
                    seen_patterns.add(pattern)
        
        for (label_u, label_v), count in edge_pattern_counts.items():
            if count >= min_count:
                sg = nx.Graph()
                sg.add_node(0, label=label_u)
                sg.add_node(1, label=label_v)
                sg.add_edge(0, 1)
                patterns.append(sg)
                pattern_supports.append(count / n_graphs)
        
        # Mine 3-node path patterns
        if self.max_subgraph_size >= 3:
            path_pattern_counts = defaultdict(int)
            for G in graphs:
                seen_patterns = set()
                for node in G.nodes():
                    neighbors = list(G.neighbors(node))
                    if len(neighbors) >= 2:
                        label_center = G.nodes[node].get('label', 0)
                        for i in range(len(neighbors)):
                            for j in range(i+1, len(neighbors)):
                                label_i = G.nodes[neighbors[i]].get('label', 0)
                                label_j = G.nodes[neighbors[j]].get('label', 0)
                                # Canonical form
                                pattern = (label_center, tuple(sorted([label_i, label_j])))
                                if pattern not in seen_patterns:
                                    path_pattern_counts[pattern] += 1
                                    seen_patterns.add(pattern)
            
            for (center, (l1, l2)), count in path_pattern_counts.items():
                if count >= min_count:
                    sg = nx.Graph()
                    sg.add_node(0, label=center)
                    sg.add_node(1, label=l1)
                    sg.add_node(2, label=l2)
                    sg.add_edge(0, 1)
                    sg.add_edge(0, 2)
                    patterns.append(sg)
                    pattern_supports.append(count / n_graphs)
        
        # Mine triangle patterns
        if self.max_subgraph_size >= 3:
            triangle_pattern_counts = defaultdict(int)
            for G in graphs:
                seen_patterns = set()
                for node in G.nodes():
                    neighbors = list(G.neighbors(node))
                    for i in range(len(neighbors)):
                        for j in range(i+1, len(neighbors)):
                            if G.has_edge(neighbors[i], neighbors[j]):
                                labels = sorted([
                                    G.nodes[node].get('label', 0),
                                    G.nodes[neighbors[i]].get('label', 0),
                                    G.nodes[neighbors[j]].get('label', 0)
                                ])
                                pattern = tuple(labels)
                                if pattern not in seen_patterns:
                                    triangle_pattern_counts[pattern] += 1
                                    seen_patterns.add(pattern)
            
            for pattern, count in triangle_pattern_counts.items():
                if count >= min_count:
                    sg = nx.Graph()
                    for i, label in enumerate(pattern):
                        sg.add_node(i, label=label)
                    sg.add_edge(0, 1)
                    sg.add_edge(1, 2)
                    sg.add_edge(0, 2)
                    patterns.append(sg)
                    pattern_supports.append(count / n_graphs)
        
        self.frequent_subgraphs = patterns
        self.pattern_supports = pattern_supports
        self.mining_time = time.time() - start_time
        
        print(f"Mined {len(patterns)} frequent subgraphs in {self.mining_time:.4f}s")
        print(f"  - Min support: {self.min_support} ({min_count} graphs)")
        print(f"  - Pattern sizes: 1-node={sum(1 for p in patterns if p.number_of_nodes()==1)}, "
              f"2-node={sum(1 for p in patterns if p.number_of_nodes()==2)}, "
              f"3-node={sum(1 for p in patterns if p.number_of_nodes()==3)}")
        
        return patterns
    
    def get_subgraph_names(self):
        """Get descriptive names for mined subgraphs."""
        names = []
        for i, sg in enumerate(self.frequent_subgraphs):
            if sg.number_of_nodes() == 1:
                label = sg.nodes[0].get('label', 0)
                names.append(f"Node({label})")
            elif sg.number_of_nodes() == 2:
                l0 = sg.nodes[0].get('label', 0)
                l1 = sg.nodes[1].get('label', 0)
                names.append(f"Edge({l0}-{l1})")
            else:
                labels = [sg.nodes[n].get('label', 0) for n in sg.nodes()]
                n_edges = sg.number_of_edges()
                if n_edges == 2:
                    names.append(f"Star({labels})")
                else:
                    names.append(f"Tri({labels})")
        return names


def count_subgraph_occurrences(graph, subgraph):
    """
    Count occurrences of a subgraph pattern in a graph.
    Uses node label matching.
    
    Args:
        graph: NetworkX graph to search in
        subgraph: NetworkX subgraph pattern to find
        
    Returns:
        Number of occurrences (0 or 1 for presence, or actual count)
    """
    if subgraph.number_of_nodes() == 1:
        # Single node pattern - count nodes with matching label
        target_label = subgraph.nodes[0].get('label', 0)
        count = sum(1 for n in graph.nodes() 
                   if graph.nodes[n].get('label', 0) == target_label)
        return count
    
    # Use subgraph isomorphism for larger patterns
    node_match = isomorphism.categorical_node_match('label', 0)
    GM = isomorphism.GraphMatcher(graph, subgraph, node_match=node_match)
    
    # Count number of matches (presence = 1 if any match exists)
    matches = list(GM.subgraph_isomorphisms_iter())
    return len(matches)


def construct_feature_vectors(graphs, frequent_subgraphs, binary=False):
    """
    Construct feature vectors based on subgraph counts.
    
    Args:
        graphs: List of NetworkX graphs
        frequent_subgraphs: List of frequent subgraph patterns
        binary: If True, use binary features (presence/absence)
        
    Returns:
        numpy array of shape (n_graphs, n_patterns)
    """
    n_graphs = len(graphs)
    n_patterns = len(frequent_subgraphs)
    
    features = np.zeros((n_graphs, n_patterns))
    
    for i, G in enumerate(graphs):
        for j, pattern in enumerate(frequent_subgraphs):
            count = count_subgraph_occurrences(G, pattern)
            features[i, j] = 1 if (binary and count > 0) else count
    
    return features


def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive evaluation metrics including imbalance-aware metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = np.nan
    
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test, **kwargs):
    """Train and evaluate Random Forest classifier."""
    params = {
        'n_estimators': kwargs.get('n_estimators', 100),
        'max_depth': kwargs.get('max_depth', None),
        'min_samples_split': kwargs.get('min_samples_split', 2),
        'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
        'random_state': 42
    }
    
    clf = RandomForestClassifier(**params)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['train_time'] = train_time
    metrics['inference_time'] = inference_time
    
    return clf, metrics


def train_svm(X_train, y_train, X_test, y_test, **kwargs):
    """Train and evaluate SVM classifier."""
    params = {
        'C': kwargs.get('C', 1.0),
        'kernel': kwargs.get('kernel', 'rbf'),
        'gamma': kwargs.get('gamma', 'scale'),
        'probability': True,
        'random_state': 42
    }
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = SVC(**params)
    
    start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    inference_time = time.time() - start_time
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['train_time'] = train_time
    metrics['inference_time'] = inference_time
    
    return clf, metrics, scaler


def train_gradient_boosting(X_train, y_train, X_test, y_test, **kwargs):
    """Train and evaluate Gradient Boosting classifier."""
    params = {
        'n_estimators': kwargs.get('n_estimators', 100),
        'max_depth': kwargs.get('max_depth', 3),
        'learning_rate': kwargs.get('learning_rate', 0.1),
        'random_state': 42
    }
    
    clf = GradientBoostingClassifier(**params)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['train_time'] = train_time
    metrics['inference_time'] = inference_time
    
    return clf, metrics


def ablation_support_threshold(graphs, labels, train_idx, test_idx,
                               thresholds=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]):
    """
    Ablation study on mining support threshold.
    
    Args:
        graphs: List of NetworkX graphs
        labels: List of graph labels
        train_idx, test_idx: Train/test indices
        thresholds: List of support thresholds to test
        
    Returns:
        DataFrame with results
    """
    results = []
    
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    
    for threshold in thresholds:
        print(f"\n--- Testing support threshold: {threshold} ---")
        
        # Mine subgraphs
        miner = GSpanMiner(min_support=threshold)
        with Timer("Mining") as mining_timer:
            patterns = miner.mine(train_graphs)
        
        if len(patterns) == 0:
            print("No patterns found, skipping...")
            continue
        
        # Construct features
        with Timer("Feature extraction") as feat_timer:
            X_train = construct_feature_vectors(train_graphs, patterns)
            X_test = construct_feature_vectors(test_graphs, patterns)
        
        # Train Random Forest
        rf_clf, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
        
        results.append({
            'threshold': threshold,
            'n_patterns': len(patterns),
            'mining_time': mining_timer.elapsed,
            'feature_time': feat_timer.elapsed,
            'rf_accuracy': rf_metrics['accuracy'],
            'rf_f1': rf_metrics['f1'],
            'rf_auc': rf_metrics['auc'],
            'rf_train_time': rf_metrics['train_time'],
        })
    
    return pd.DataFrame(results)


def ablation_model_parameters(X_train, y_train, X_test, y_test):
    """
    Ablation study on model parameters.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with ablation results for each model
    """
    results = {'random_forest': [], 'svm': [], 'gradient_boosting': []}
    
    # Random Forest ablation
    print("\n=== Random Forest Ablation ===")
    rf_params = {
        'n_estimators': [10, 50, 100, 200],
        'max_depth': [3, 5, 10, None],
    }
    
    for n_est in rf_params['n_estimators']:
        for max_d in rf_params['max_depth']:
            clf, metrics = train_random_forest(
                X_train, y_train, X_test, y_test,
                n_estimators=n_est, max_depth=max_d
            )
            results['random_forest'].append({
                'n_estimators': n_est,
                'max_depth': max_d if max_d else 'None',
                **metrics
            })
    
    # SVM ablation
    print("\n=== SVM Ablation ===")
    svm_params = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf', 'poly'],
    }
    
    for C in svm_params['C']:
        for kernel in svm_params['kernel']:
            try:
                clf, metrics, _ = train_svm(
                    X_train, y_train, X_test, y_test,
                    C=C, kernel=kernel
                )
                results['svm'].append({
                    'C': C,
                    'kernel': kernel,
                    **metrics
                })
            except Exception as e:
                print(f"SVM failed with C={C}, kernel={kernel}: {e}")
    
    # Gradient Boosting ablation
    print("\n=== Gradient Boosting Ablation ===")
    gb_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
    }
    
    for n_est in gb_params['n_estimators']:
        for lr in gb_params['learning_rate']:
            clf, metrics = train_gradient_boosting(
                X_train, y_train, X_test, y_test,
                n_estimators=n_est, learning_rate=lr
            )
            results['gradient_boosting'].append({
                'n_estimators': n_est,
                'learning_rate': lr,
                **metrics
            })
    
    return {k: pd.DataFrame(v) for k, v in results.items()}


def get_feature_importance(clf, feature_names, model_type='rf'):
    """
    Get feature importance from trained classifier.
    
    Args:
        clf: Trained classifier
        feature_names: List of feature names
        model_type: 'rf' for Random Forest, 'svm' for SVM
        
    Returns:
        DataFrame with feature importances
    """
    if model_type == 'rf' or model_type == 'gb':
        importances = clf.feature_importances_
    elif model_type == 'svm':
        # For linear SVM, use coefficient magnitudes
        if clf.kernel == 'linear':
            importances = np.abs(clf.coef_[0])
        else:
            # For non-linear SVM, return None
            return None
    else:
        return None
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df


def run_q1_experiments(save_dir='./results/q1'):
    """
    Run all Q1 experiments: Frequent Subgraph Mining + Classic ML
    
    Args:
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Q1: FREQUENT SUBGRAPH MINING + CLASSIC ML")
    print("=" * 70)
    
    # Load dataset
    dataset, info = load_mutag_dataset()
    graphs, labels = pyg_to_networkx_graphs(dataset)
    
    # Analyze class distribution
    class_counts = Counter(labels)
    print(f"\nClass distribution: {class_counts}")
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain: {len(train_idx)}, Test: {len(test_idx)}")
    
    # =====================================================
    # Experiment 1: Ablation on Support Threshold
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 1: Support Threshold Ablation")
    print("=" * 60)
    
    threshold_results = ablation_support_threshold(
        graphs, labels, train_idx, test_idx,
        thresholds=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    )
    
    print("\nSupport Threshold Results:")
    print(threshold_results.to_string(index=False))
    threshold_results.to_csv(f'{save_dir}/threshold_ablation.csv', index=False)
    
    # =====================================================
    # Experiment 2: Best Configuration Training
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 2: Full Model Training (Support=0.1)")
    print("=" * 60)
    
    # Use best threshold from ablation
    best_threshold = 0.1
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    
    # Mine frequent subgraphs
    miner = GSpanMiner(min_support=best_threshold)
    patterns = miner.mine(train_graphs)
    feature_names = miner.get_subgraph_names()
    
    # Construct feature vectors
    print("\nConstructing feature vectors...")
    X_train = construct_feature_vectors(train_graphs, patterns)
    X_test = construct_feature_vectors(test_graphs, patterns)
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Train models
    print("\n--- Training Random Forest ---")
    rf_clf, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    print(f"Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {rf_metrics['balanced_accuracy']:.4f}")
    print(f"F1-Score (macro): {rf_metrics['f1_macro']:.4f}")
    print(f"AUC: {rf_metrics['auc']:.4f}")
    
    print("\n--- Training SVM ---")
    svm_clf, svm_metrics, svm_scaler = train_svm(
        X_train, y_train, X_test, y_test,
        kernel='rbf', C=1.0
    )
    print(f"Accuracy: {svm_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {svm_metrics['balanced_accuracy']:.4f}")
    print(f"F1-Score (macro): {svm_metrics['f1_macro']:.4f}")
    print(f"AUC: {svm_metrics['auc']:.4f}")
    
    print("\n--- Training Gradient Boosting ---")
    gb_clf, gb_metrics = train_gradient_boosting(X_train, y_train, X_test, y_test)
    print(f"Accuracy: {gb_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {gb_metrics['balanced_accuracy']:.4f}")
    print(f"F1-Score (macro): {gb_metrics['f1_macro']:.4f}")
    print(f"AUC: {gb_metrics['auc']:.4f}")
    
    # =====================================================
    # Experiment 3: Imbalance Handling Strategies
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 3: Imbalance Handling Strategies")
    print("=" * 60)
    
    imbalance_results = []
    
    # Baseline (no handling)
    print("\n--- Baseline (No Imbalance Handling) ---")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['strategy'] = 'Baseline'
    imbalance_results.append(metrics)
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")
    
    # Class Weights
    print("\n--- Class Weighting ---")
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['strategy'] = 'Class Weights'
    imbalance_results.append(metrics)
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")
    
    # SMOTE (if available)
    if IMBLEARN_AVAILABLE:
        print("\n--- SMOTE Oversampling ---")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"  Original: {Counter(y_train)}, After SMOTE: {Counter(y_train_smote)}")
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_smote, y_train_smote)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['strategy'] = 'SMOTE'
        imbalance_results.append(metrics)
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")
        
        # Random Undersampling
        print("\n--- Random Undersampling ---")
        rus = RandomUnderSampler(random_state=42)
        X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
        print(f"  Original: {Counter(y_train)}, After Undersampling: {Counter(y_train_under)}")
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_under, y_train_under)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['strategy'] = 'Undersampling'
        imbalance_results.append(metrics)
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")
        
        # SMOTEENN (Hybrid)
        print("\n--- SMOTEENN (Hybrid) ---")
        smoteenn = SMOTEENN(random_state=42)
        X_train_se, y_train_se = smoteenn.fit_resample(X_train, y_train)
        print(f"  Original: {Counter(y_train)}, After SMOTEENN: {Counter(y_train_se)}")
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_se, y_train_se)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['strategy'] = 'SMOTEENN'
        imbalance_results.append(metrics)
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")
    
    imbalance_df = pd.DataFrame(imbalance_results)
    print("\n" + "=" * 40)
    print("Imbalance Handling Comparison:")
    cols = ['strategy', 'accuracy', 'balanced_accuracy', 'f1_macro', 'mcc']
    if 'auc' in imbalance_df.columns:
        cols.append('auc')
    print(imbalance_df[cols].to_string(index=False))
    imbalance_df.to_csv(f'{save_dir}/imbalance_handling.csv', index=False)
    
    # =====================================================
    # Experiment 4: Model Parameter Ablation
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 4: Model Parameter Ablation")
    print("=" * 60)
    
    param_results = ablation_model_parameters(X_train, y_train, X_test, y_test)
    
    for model_name, df in param_results.items():
        print(f"\n{model_name.upper()} Results:")
        print(df.to_string(index=False))
        df.to_csv(f'{save_dir}/{model_name}_ablation.csv', index=False)
    
    # =====================================================
    # Feature Importance Analysis
    # =====================================================
    print("\n" + "=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)
    
    rf_importance = get_feature_importance(rf_clf, feature_names, 'rf')
    print("\nTop 10 Important Features (Random Forest):")
    print(rf_importance.head(10).to_string(index=False))
    rf_importance.to_csv(f'{save_dir}/rf_feature_importance.csv', index=False)
    
    # =====================================================
    # Experiment 5: Cross-Validation
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 5: 5-Fold Cross-Validation")
    print("=" * 60)
    
    # Use all data for CV
    all_graphs = graphs
    all_labels = labels
    
    # Mine patterns on all data for CV (or could do nested CV)
    miner_cv = GSpanMiner(min_support=0.1)
    patterns_cv = miner_cv.mine(all_graphs)
    X_all = construct_feature_vectors(all_graphs, patterns_cv)
    y_all = np.array(all_labels)
    
    cv_results = cross_validate_all_classic_ml(X_all, y_all, n_folds=5)
    cv_results.to_csv(f'{save_dir}/cv_comparison.csv', index=False)
    
    # Save final summary
    summary = {
        'Model': ['Random Forest', 'SVM (RBF)', 'Gradient Boosting'],
        'Accuracy': [rf_metrics['accuracy'], svm_metrics['accuracy'], gb_metrics['accuracy']],
        'Balanced_Accuracy': [rf_metrics['balanced_accuracy'], svm_metrics['balanced_accuracy'], gb_metrics['balanced_accuracy']],
        'Precision': [rf_metrics['precision'], svm_metrics['precision'], gb_metrics['precision']],
        'Recall': [rf_metrics['recall'], svm_metrics['recall'], gb_metrics['recall']],
        'F1': [rf_metrics['f1'], svm_metrics['f1'], gb_metrics['f1']],
        'F1_Macro': [rf_metrics['f1_macro'], svm_metrics['f1_macro'], gb_metrics['f1_macro']],
        'MCC': [rf_metrics['mcc'], svm_metrics['mcc'], gb_metrics['mcc']],
        'AUC': [rf_metrics['auc'], svm_metrics['auc'], gb_metrics['auc']],
        'Train Time': [rf_metrics['train_time'], svm_metrics['train_time'], gb_metrics['train_time']],
        'Inference Time': [rf_metrics['inference_time'], svm_metrics['inference_time'], gb_metrics['inference_time']],
    }
    
    summary_df = pd.DataFrame(summary)
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - Q1")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f'{save_dir}/q1_summary.csv', index=False)
    
    print("\nCross-Validation Results (5-fold):")
    cv_display_cols = ['model', 'accuracy_mean', 'accuracy_std', 'balanced_accuracy_mean']
    available_cols = [c for c in cv_display_cols if c in cv_results.columns]
    print(cv_results[available_cols].to_string(index=False))
    
    # Find best imbalance handling strategy
    if len(imbalance_results) > 0:
        best_strategy = imbalance_df.loc[imbalance_df['balanced_accuracy'].idxmax()]
        print(f"\nBest Imbalance Handling Strategy: {best_strategy['strategy']}")
        print(f"  Balanced Accuracy: {best_strategy['balanced_accuracy']:.4f}")
    
    return {
        'patterns': patterns,
        'feature_names': feature_names,
        'rf_clf': rf_clf,
        'svm_clf': svm_clf,
        'gb_clf': gb_clf,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'threshold_results': threshold_results,
        'param_results': param_results,
        'imbalance_results': imbalance_df,
        'cv_results': cv_results,
        'summary': summary_df
    }


if __name__ == "__main__":
    results = run_q1_experiments()
