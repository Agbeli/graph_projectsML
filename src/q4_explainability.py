"""
q4_explainability.py - Q4: GNN Explainability Analysis

This module implements:
1. GNNExplainer for post-hoc explanations
2. Evaluation metrics: Fidelity+, Fidelity-, Sparsity, Runtime
3. Comparison with Classic ML interpretability (feature importance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import to_networkx
import networkx as nx
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from data_utils import load_mutag_dataset, pyg_to_networkx_graphs, split_dataset, get_data_loaders
from q1_gspan_classic_ml import (
    GSpanMiner, construct_feature_vectors,
    train_random_forest, train_svm, get_feature_importance
)
from q2_gnn_models import get_model, train_model, evaluate, device


class GNNExplainerWrapper:
    """
    Wrapper for GNNExplainer with custom evaluation metrics.
    """
    
    def __init__(self, model, num_features, edge_mask_type='object'):
        """
        Args:
            model: Trained GNN model
            num_features: Number of node features
            edge_mask_type: Type of edge mask ('object' or 'sigmoid')
        """
        self.model = model
        self.model.eval()
        
        # Initialize explainer
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200, lr=0.01),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            )
        )
        
    def explain_graph(self, data):
        """
        Generate explanation for a single graph.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Dictionary containing node_mask, edge_mask, and prediction
        """
        data = data.to(device)
        
        # Get original prediction
        with torch.no_grad():
            original_out = self.model(data.x, data.edge_index, 
                                     torch.zeros(data.num_nodes, dtype=torch.long, device=device))
            original_pred = original_out.argmax(dim=1).item()
            original_prob = F.softmax(original_out, dim=1)[0, original_pred].item()
        
        # Generate explanation
        start_time = time.time()
        explanation = self.explainer(data.x, data.edge_index, 
                                     batch=torch.zeros(data.num_nodes, dtype=torch.long, device=device))
        explain_time = time.time() - start_time
        
        return {
            'node_mask': explanation.node_mask.detach().cpu().numpy() if explanation.node_mask is not None else None,
            'edge_mask': explanation.edge_mask.detach().cpu().numpy() if explanation.edge_mask is not None else None,
            'prediction': original_pred,
            'probability': original_prob,
            'explain_time': explain_time
        }


def compute_fidelity_plus(model, data, node_mask, edge_mask, top_k=0.5):
    """
    Compute Fidelity+ (faithfulness of important features).
    Higher is better - measures if keeping important features maintains prediction.
    
    Args:
        model: GNN model
        data: Original graph data
        node_mask: Node importance mask
        edge_mask: Edge importance mask
        top_k: Fraction of top features to keep
        
    Returns:
        Fidelity+ score
    """
    model.eval()
    data = data.to(device)
    
    # Original prediction
    with torch.no_grad():
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        original_out = model(data.x, data.edge_index, batch)
        original_prob = F.softmax(original_out, dim=1)
        original_pred = original_out.argmax(dim=1).item()
        original_conf = original_prob[0, original_pred].item()
    
    # Keep only top-k important edges
    if edge_mask is not None and len(edge_mask) > 0:
        k = max(1, int(top_k * len(edge_mask)))
        top_indices = np.argsort(edge_mask)[-k:]
        
        # Create masked edge index
        mask = torch.zeros(len(edge_mask), dtype=torch.bool, device=device)
        mask[top_indices] = True
        masked_edge_index = data.edge_index[:, mask]
        
        # Prediction with only important edges
        with torch.no_grad():
            masked_out = model(data.x, masked_edge_index, batch)
            masked_prob = F.softmax(masked_out, dim=1)
            masked_conf = masked_prob[0, original_pred].item()
        
        # Fidelity+ = probability of original class with important features
        return masked_conf
    
    return original_conf


def compute_fidelity_minus(model, data, node_mask, edge_mask, top_k=0.5):
    """
    Compute Fidelity- (faithfulness when removing important features).
    Higher is better - measures if removing important features changes prediction.
    
    Args:
        model: GNN model
        data: Original graph data
        node_mask: Node importance mask
        edge_mask: Edge importance mask  
        top_k: Fraction of top features to remove
        
    Returns:
        Fidelity- score (original_prob - removed_prob)
    """
    model.eval()
    data = data.to(device)
    
    # Original prediction
    with torch.no_grad():
        batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        original_out = model(data.x, data.edge_index, batch)
        original_prob = F.softmax(original_out, dim=1)
        original_pred = original_out.argmax(dim=1).item()
        original_conf = original_prob[0, original_pred].item()
    
    # Remove top-k important edges
    if edge_mask is not None and len(edge_mask) > 0:
        k = max(1, int(top_k * len(edge_mask)))
        top_indices = np.argsort(edge_mask)[-k:]
        
        # Create masked edge index (remove important edges)
        mask = torch.ones(len(edge_mask), dtype=torch.bool, device=device)
        mask[top_indices] = False
        masked_edge_index = data.edge_index[:, mask]
        
        if masked_edge_index.size(1) == 0:
            # All edges removed
            return original_conf
        
        # Prediction without important edges
        with torch.no_grad():
            masked_out = model(data.x, masked_edge_index, batch)
            masked_prob = F.softmax(masked_out, dim=1)
            masked_conf = masked_prob[0, original_pred].item()
        
        # Fidelity- = drop in confidence when removing important features
        return original_conf - masked_conf
    
    return 0.0


def compute_sparsity(edge_mask, threshold=0.5):
    """
    Compute sparsity of explanation.
    Higher sparsity means more focused explanation.
    
    Args:
        edge_mask: Edge importance mask
        threshold: Threshold for considering an edge as important
        
    Returns:
        Sparsity score (fraction of edges below threshold)
    """
    if edge_mask is None or len(edge_mask) == 0:
        return 0.0
    
    # Normalize mask
    mask_normalized = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min() + 1e-8)
    
    # Sparsity = fraction of edges with low importance
    sparsity = np.mean(mask_normalized < threshold)
    return sparsity


def evaluate_explanations(model, dataset, n_samples=50, seed=42):
    """
    Evaluate explanation quality on multiple graphs.
    
    Args:
        model: Trained GNN model
        dataset: PyTorch Geometric dataset
        n_samples: Number of graphs to explain
        seed: Random seed
        
    Returns:
        DataFrame with evaluation metrics
    """
    np.random.seed(seed)
    
    # Sample graphs
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    # Initialize explainer
    explainer_wrapper = GNNExplainerWrapper(model, dataset.num_node_features)
    
    results = []
    
    print(f"Evaluating explanations for {len(indices)} graphs...")
    
    for i, idx in enumerate(indices):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(indices)}...")
        
        data = dataset[idx]
        
        try:
            # Generate explanation
            explanation = explainer_wrapper.explain_graph(data)
            
            # Compute metrics
            fidelity_plus = compute_fidelity_plus(
                model, data, 
                explanation['node_mask'], 
                explanation['edge_mask']
            )
            
            fidelity_minus = compute_fidelity_minus(
                model, data,
                explanation['node_mask'],
                explanation['edge_mask']
            )
            
            sparsity = compute_sparsity(explanation['edge_mask'])
            
            results.append({
                'graph_idx': idx,
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'prediction': explanation['prediction'],
                'confidence': explanation['probability'],
                'fidelity_plus': fidelity_plus,
                'fidelity_minus': fidelity_minus,
                'sparsity': sparsity,
                'runtime': explanation['explain_time']
            })
            
        except Exception as e:
            print(f"  Error explaining graph {idx}: {e}")
            continue
    
    return pd.DataFrame(results)


def compare_with_classic_ml(dataset, save_dir):
    """
    Compare GNN explanations with Classic ML interpretability.
    
    Args:
        dataset: PyTorch Geometric dataset
        save_dir: Directory to save results
    """
    print("\n" + "=" * 60)
    print("CLASSIC ML INTERPRETABILITY")
    print("=" * 60)
    
    # Prepare data
    graphs, labels = pyg_to_networkx_graphs(dataset)
    indices = np.arange(len(graphs))
    train_idx, test_idx = indices[:150], indices[150:]
    
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    
    # Mine patterns and create features
    miner = GSpanMiner(min_support=0.1)
    patterns = miner.mine(train_graphs)
    feature_names = miner.get_subgraph_names()
    
    X_train = construct_feature_vectors(train_graphs, patterns)
    X_test = construct_feature_vectors(test_graphs, patterns)
    
    # Train Random Forest
    rf_clf, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Get feature importance
    rf_importance = get_feature_importance(rf_clf, feature_names, 'rf')
    
    print("\nTop 10 Important Subgraph Features (Random Forest):")
    print(rf_importance.head(10).to_string(index=False))
    
    # Save feature importance
    rf_importance.to_csv(f'{save_dir}/classic_ml_feature_importance.csv', index=False)
    
    return rf_importance, feature_names


def create_explainability_plots(eval_results, feature_importance, save_dir):
    """Create visualization plots for explainability analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Fidelity+ Distribution
    ax1 = axes[0, 0]
    ax1.hist(eval_results['fidelity_plus'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(eval_results['fidelity_plus'].mean(), color='red', linestyle='--', 
                label=f'Mean: {eval_results["fidelity_plus"].mean():.3f}')
    ax1.set_xlabel('Fidelity+')
    ax1.set_ylabel('Count')
    ax1.set_title('Fidelity+ Distribution (Higher is Better)')
    ax1.legend()
    
    # Plot 2: Fidelity- Distribution
    ax2 = axes[0, 1]
    ax2.hist(eval_results['fidelity_minus'], bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(eval_results['fidelity_minus'].mean(), color='red', linestyle='--',
                label=f'Mean: {eval_results["fidelity_minus"].mean():.3f}')
    ax2.set_xlabel('Fidelity-')
    ax2.set_ylabel('Count')
    ax2.set_title('Fidelity- Distribution (Higher is Better)')
    ax2.legend()
    
    # Plot 3: Sparsity Distribution
    ax3 = axes[1, 0]
    ax3.hist(eval_results['sparsity'], bins=20, color='seagreen', alpha=0.7, edgecolor='black')
    ax3.axvline(eval_results['sparsity'].mean(), color='red', linestyle='--',
                label=f'Mean: {eval_results["sparsity"].mean():.3f}')
    ax3.set_xlabel('Sparsity')
    ax3.set_ylabel('Count')
    ax3.set_title('Sparsity Distribution')
    ax3.legend()
    
    # Plot 4: Top Feature Importance (Classic ML)
    ax4 = axes[1, 1]
    top_features = feature_importance.head(10)
    ax4.barh(range(len(top_features)), top_features['importance'], color='purple', alpha=0.7)
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['feature'])
    ax4.set_xlabel('Importance')
    ax4.set_title('Top 10 Subgraph Features (Classic ML)')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/explainability_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Explainability plots saved to {save_dir}/explainability_plots.png")


def run_q4_experiments(save_dir='./results/q4'):
    """
    Run all Q4 experiments: Explainability Analysis.
    
    Args:
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Q4: EXPLAINABILITY ANALYSIS")
    print("=" * 70)
    
    # Load dataset
    dataset, info = load_mutag_dataset()
    
    # Split data
    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size=32
    )
    
    # =====================================================
    # Train GNN Model for Explanation
    # =====================================================
    print("\n" + "=" * 60)
    print("TRAINING GNN MODEL")
    print("=" * 60)
    
    # Use GIN as it often provides better explanations
    model = get_model(
        'GIN',
        num_features=dataset.num_node_features,
        hidden_dim=64,
        num_classes=dataset.num_classes,
        num_layers=3
    )
    
    model, history = train_model(
        model, train_loader, val_loader,
        epochs=200, lr=0.001, verbose=True
    )
    
    test_metrics, _, _, _ = evaluate(model, test_loader)
    print(f"\nGIN Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # =====================================================
    # GNNExplainer Evaluation
    # =====================================================
    print("\n" + "=" * 60)
    print("GNNExplainer EVALUATION")
    print("=" * 60)
    
    eval_results = evaluate_explanations(model, dataset, n_samples=50)
    
    print("\nExplanation Quality Metrics (GNNExplainer):")
    print(f"  Fidelity+:  {eval_results['fidelity_plus'].mean():.4f} ± {eval_results['fidelity_plus'].std():.4f}")
    print(f"  Fidelity-:  {eval_results['fidelity_minus'].mean():.4f} ± {eval_results['fidelity_minus'].std():.4f}")
    print(f"  Sparsity:   {eval_results['sparsity'].mean():.4f} ± {eval_results['sparsity'].std():.4f}")
    print(f"  Runtime:    {eval_results['runtime'].mean():.4f} ± {eval_results['runtime'].std():.4f} seconds")
    
    eval_results.to_csv(f'{save_dir}/gnnexplainer_results.csv', index=False)
    
    # =====================================================
    # Classic ML Interpretability
    # =====================================================
    feature_importance, feature_names = compare_with_classic_ml(dataset, save_dir)
    
    # =====================================================
    # Comparison Summary
    # =====================================================
    print("\n" + "=" * 60)
    print("EXPLAINABILITY COMPARISON")
    print("=" * 60)
    
    comparison_summary = """
COMPARISON: GNN EXPLANATIONS vs CLASSIC ML INTERPRETABILITY

1. GNNExplainer (Post-hoc, Local)
   - Provides instance-level explanations
   - Identifies important edges/nodes for each prediction
   - Fidelity metrics measure faithfulness to model behavior
   - Runtime per explanation: ~{:.3f}s
   - Pros: Works with any GNN architecture, provides visual explanations
   - Cons: Computationally expensive, explanations may vary

2. Classic ML with Frequent Subgraphs (Self-Explainable, Global)
   - Feature importance directly from model (e.g., Random Forest)
   - Subgraph patterns provide semantic meaning
   - Global interpretation: same features for all predictions
   - Pros: Fast, consistent, chemically meaningful patterns
   - Cons: Limited expressiveness, requires feature engineering

3. Key Differences:
   - Scope: GNNExplainer=local (per-graph), Classic ML=global (dataset-level)
   - Granularity: GNNExplainer=node/edge level, Classic ML=subgraph level
   - Computation: GNNExplainer=expensive, Classic ML=cheap (after training)
   - Interpretability: Classic ML more intuitive for domain experts

4. Recommendations:
   - For debugging: Use GNNExplainer to understand individual predictions
   - For domain insights: Use Classic ML feature importance
   - For deployment: Consider hybrid approach with interpretable GNNs
""".format(eval_results['runtime'].mean())
    
    print(comparison_summary)
    
    # Save comparison summary
    with open(f'{save_dir}/comparison_summary.txt', 'w') as f:
        f.write(comparison_summary)
    
    # Create summary statistics
    summary_metrics = {
        'Metric': ['Fidelity+', 'Fidelity-', 'Sparsity', 'Runtime (s)'],
        'Mean': [
            eval_results['fidelity_plus'].mean(),
            eval_results['fidelity_minus'].mean(),
            eval_results['sparsity'].mean(),
            eval_results['runtime'].mean()
        ],
        'Std': [
            eval_results['fidelity_plus'].std(),
            eval_results['fidelity_minus'].std(),
            eval_results['sparsity'].std(),
            eval_results['runtime'].std()
        ],
        'Min': [
            eval_results['fidelity_plus'].min(),
            eval_results['fidelity_minus'].min(),
            eval_results['sparsity'].min(),
            eval_results['runtime'].min()
        ],
        'Max': [
            eval_results['fidelity_plus'].max(),
            eval_results['fidelity_minus'].max(),
            eval_results['sparsity'].max(),
            eval_results['runtime'].max()
        ]
    }
    
    summary_df = pd.DataFrame(summary_metrics)
    print("\nGNNExplainer Summary Statistics:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f'{save_dir}/explainability_summary.csv', index=False)
    
    # Create visualizations
    create_explainability_plots(eval_results, feature_importance, save_dir)
    
    return {
        'eval_results': eval_results,
        'feature_importance': feature_importance,
        'summary': summary_df,
        'model': model
    }


if __name__ == "__main__":
    results = run_q4_experiments()
