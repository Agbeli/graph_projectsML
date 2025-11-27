"""
q3_comparison.py - Q3: Comparison of Classic ML vs GNN Approaches

This module compares:
1. Classification quality metrics (accuracy, precision, recall, F1, AUC)
2. Training and inference efficiency
3. Scalability and complexity analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Import local modules
from data_utils import load_mutag_dataset, pyg_to_networkx_graphs, split_dataset, get_data_loaders
from q1_gspan_classic_ml import (
    GSpanMiner, construct_feature_vectors, 
    train_random_forest, train_svm, train_gradient_boosting
)
from q2_gnn_models import get_model, train_model, evaluate, device


def comprehensive_comparison(save_dir='./results/q3'):
    """
    Perform comprehensive comparison between Classic ML and GNN approaches.
    
    Args:
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Q3: COMPARISON - CLASSIC ML vs GNN")
    print("=" * 70)
    
    # Load dataset
    dataset, info = load_mutag_dataset()
    graphs, labels = pyg_to_networkx_graphs(dataset)
    
    # Split data
    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size=32
    )
    
    # Prepare classic ML data
    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    val_graphs = [graphs[i] for i in val_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]
    y_val = [labels[i] for i in val_idx]
    
    results = []
    timing_results = []
    
    # =====================================================
    # Classic ML: Frequent Subgraph Mining
    # =====================================================
    print("\n" + "=" * 60)
    print("CLASSIC ML APPROACH")
    print("=" * 60)
    
    # Mine frequent subgraphs
    miner = GSpanMiner(min_support=0.1)
    mining_start = time.time()
    patterns = miner.mine(train_graphs)
    mining_time = time.time() - mining_start
    
    # Feature extraction
    feature_start = time.time()
    X_train = construct_feature_vectors(train_graphs, patterns)
    X_test = construct_feature_vectors(test_graphs, patterns)
    feature_time = time.time() - feature_start
    
    total_preprocess_time = mining_time + feature_time
    
    # Random Forest
    print("\n--- Random Forest ---")
    rf_clf, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results.append({
        'Approach': 'Classic ML',
        'Model': 'Random Forest',
        'Accuracy': rf_metrics['accuracy'],
        'Precision': rf_metrics['precision'],
        'Recall': rf_metrics['recall'],
        'F1': rf_metrics['f1'],
        'AUC': rf_metrics['auc'],
        'Train Time': rf_metrics['train_time'] + total_preprocess_time,
        'Inference Time': rf_metrics['inference_time'],
    })
    print(f"Accuracy: {rf_metrics['accuracy']:.4f}, F1: {rf_metrics['f1']:.4f}")
    
    # SVM
    print("\n--- SVM (RBF) ---")
    svm_clf, svm_metrics, _ = train_svm(X_train, y_train, X_test, y_test)
    results.append({
        'Approach': 'Classic ML',
        'Model': 'SVM (RBF)',
        'Accuracy': svm_metrics['accuracy'],
        'Precision': svm_metrics['precision'],
        'Recall': svm_metrics['recall'],
        'F1': svm_metrics['f1'],
        'AUC': svm_metrics['auc'],
        'Train Time': svm_metrics['train_time'] + total_preprocess_time,
        'Inference Time': svm_metrics['inference_time'],
    })
    print(f"Accuracy: {svm_metrics['accuracy']:.4f}, F1: {svm_metrics['f1']:.4f}")
    
    # Gradient Boosting
    print("\n--- Gradient Boosting ---")
    gb_clf, gb_metrics = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results.append({
        'Approach': 'Classic ML',
        'Model': 'Gradient Boosting',
        'Accuracy': gb_metrics['accuracy'],
        'Precision': gb_metrics['precision'],
        'Recall': gb_metrics['recall'],
        'F1': gb_metrics['f1'],
        'AUC': gb_metrics['auc'],
        'Train Time': gb_metrics['train_time'] + total_preprocess_time,
        'Inference Time': gb_metrics['inference_time'],
    })
    print(f"Accuracy: {gb_metrics['accuracy']:.4f}, F1: {gb_metrics['f1']:.4f}")
    
    timing_results.append({
        'Approach': 'Classic ML',
        'Mining Time': mining_time,
        'Feature Extraction': feature_time,
        'Model Training (avg)': np.mean([rf_metrics['train_time'], 
                                         svm_metrics['train_time'],
                                         gb_metrics['train_time']]),
        'Total Preprocessing': total_preprocess_time,
    })
    
    # =====================================================
    # GNN Approach
    # =====================================================
    print("\n" + "=" * 60)
    print("GNN APPROACH")
    print("=" * 60)
    
    gnn_models = ['GCN', 'GIN', 'GraphSAGE', 'GAT']
    
    for model_name in gnn_models:
        print(f"\n--- {model_name} ---")
        
        model = get_model(
            model_name,
            num_features=dataset.num_node_features,
            hidden_dim=64,
            num_classes=dataset.num_classes,
            num_layers=3
        )
        
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=200, lr=0.001, verbose=False
        )
        
        # Test evaluation
        test_metrics, _, _, _ = evaluate(model, test_loader)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                _ = model(data.x, data.edge_index, data.batch)
        inference_time = time.time() - start_time
        
        results.append({
            'Approach': 'GNN',
            'Model': model_name,
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1': test_metrics['f1'],
            'AUC': test_metrics['auc'],
            'Train Time': history['train_time'],
            'Inference Time': inference_time,
        })
        
        print(f"Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
    
    timing_results.append({
        'Approach': 'GNN',
        'Mining Time': 0,
        'Feature Extraction': 0,
        'Model Training (avg)': np.mean([r['Train Time'] for r in results if r['Approach'] == 'GNN']),
        'Total Preprocessing': 0,
    })
    
    # =====================================================
    # Create Comparison DataFrames
    # =====================================================
    results_df = pd.DataFrame(results)
    timing_df = pd.DataFrame(timing_results)
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 70)
    print("\nClassification Quality:")
    print(results_df[['Approach', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']].to_string(index=False))
    
    print("\nEfficiency Comparison:")
    print(results_df[['Approach', 'Model', 'Train Time', 'Inference Time']].to_string(index=False))
    
    # Save results
    results_df.to_csv(f'{save_dir}/comparison_results.csv', index=False)
    timing_df.to_csv(f'{save_dir}/timing_breakdown.csv', index=False)
    
    # =====================================================
    # Statistical Summary
    # =====================================================
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)
    
    classic_results = results_df[results_df['Approach'] == 'Classic ML']
    gnn_results = results_df[results_df['Approach'] == 'GNN']
    
    summary = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Train Time', 'Inference Time'],
        'Classic ML (Best)': [
            classic_results['Accuracy'].max(),
            classic_results['Precision'].max(),
            classic_results['Recall'].max(),
            classic_results['F1'].max(),
            classic_results['AUC'].max(),
            classic_results['Train Time'].min(),
            classic_results['Inference Time'].min(),
        ],
        'GNN (Best)': [
            gnn_results['Accuracy'].max(),
            gnn_results['Precision'].max(),
            gnn_results['Recall'].max(),
            gnn_results['F1'].max(),
            gnn_results['AUC'].max(),
            gnn_results['Train Time'].min(),
            gnn_results['Inference Time'].min(),
        ],
        'Classic ML (Avg)': [
            classic_results['Accuracy'].mean(),
            classic_results['Precision'].mean(),
            classic_results['Recall'].mean(),
            classic_results['F1'].mean(),
            classic_results['AUC'].mean(),
            classic_results['Train Time'].mean(),
            classic_results['Inference Time'].mean(),
        ],
        'GNN (Avg)': [
            gnn_results['Accuracy'].mean(),
            gnn_results['Precision'].mean(),
            gnn_results['Recall'].mean(),
            gnn_results['F1'].mean(),
            gnn_results['AUC'].mean(),
            gnn_results['Train Time'].mean(),
            gnn_results['Inference Time'].mean(),
        ],
    }
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f'{save_dir}/summary_statistics.csv', index=False)
    
    # =====================================================
    # Generate Visualizations
    # =====================================================
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    create_comparison_plots(results_df, save_dir)
    
    # =====================================================
    # Key Findings
    # =====================================================
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    best_classic = classic_results.loc[classic_results['Accuracy'].idxmax()]
    best_gnn = gnn_results.loc[gnn_results['Accuracy'].idxmax()]
    
    print(f"\n1. Best Classic ML Model: {best_classic['Model']}")
    print(f"   Accuracy: {best_classic['Accuracy']:.4f}, F1: {best_classic['F1']:.4f}")
    
    print(f"\n2. Best GNN Model: {best_gnn['Model']}")
    print(f"   Accuracy: {best_gnn['Accuracy']:.4f}, F1: {best_gnn['F1']:.4f}")
    
    accuracy_diff = best_gnn['Accuracy'] - best_classic['Accuracy']
    print(f"\n3. Accuracy Difference (GNN - Classic): {accuracy_diff:+.4f}")
    
    print(f"\n4. Training Time Comparison:")
    print(f"   Classic ML (avg): {classic_results['Train Time'].mean():.4f}s")
    print(f"   GNN (avg): {gnn_results['Train Time'].mean():.4f}s")
    
    print(f"\n5. Inference Time Comparison:")
    print(f"   Classic ML (avg): {classic_results['Inference Time'].mean():.6f}s")
    print(f"   GNN (avg): {gnn_results['Inference Time'].mean():.6f}s")
    
    return {
        'results': results_df,
        'timing': timing_df,
        'summary': summary_df
    }


def create_comparison_plots(results_df, save_dir):
    """Create visualization plots for comparison."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    classic_acc = results_df[results_df['Approach'] == 'Classic ML'][['Model', 'Accuracy']]
    gnn_acc = results_df[results_df['Approach'] == 'GNN'][['Model', 'Accuracy']]
    
    x = np.arange(max(len(classic_acc), len(gnn_acc)))
    width = 0.35
    
    ax1.bar(np.arange(len(classic_acc)), classic_acc['Accuracy'], width, 
            label='Classic ML', color='steelblue', alpha=0.8)
    ax1.bar(np.arange(len(gnn_acc)) + width + 0.1, gnn_acc['Accuracy'], width,
            label='GNN', color='coral', alpha=0.8)
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_xticks(np.arange(max(len(classic_acc), len(gnn_acc))) + width/2)
    all_models = list(classic_acc['Model']) + list(gnn_acc['Model'])
    ax1.set_xticklabels(all_models[:max(len(classic_acc), len(gnn_acc))], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: F1 Score Comparison
    ax2 = axes[0, 1]
    colors = ['steelblue' if a == 'Classic ML' else 'coral' 
              for a in results_df['Approach']]
    bars = ax2.barh(results_df['Model'], results_df['F1'], color=colors, alpha=0.8)
    ax2.set_xlabel('F1 Score')
    ax2.set_title('F1 Score by Model')
    ax2.set_xlim(0, 1.1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', alpha=0.8, label='Classic ML'),
                       Patch(facecolor='coral', alpha=0.8, label='GNN')]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    # Plot 3: Training Time Comparison
    ax3 = axes[1, 0]
    ax3.bar(results_df['Model'], results_df['Train Time'], 
            color=colors, alpha=0.8)
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    
    # Plot 4: Radar Chart for Quality Metrics
    ax4 = axes[1, 1]
    
    # Aggregate metrics by approach
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    classic_avg = results_df[results_df['Approach'] == 'Classic ML'][metrics].mean()
    gnn_avg = results_df[results_df['Approach'] == 'GNN'][metrics].mean()
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x_pos - width/2, classic_avg, width, label='Classic ML', 
            color='steelblue', alpha=0.8)
    ax4.bar(x_pos + width/2, gnn_avg, width, label='GNN', 
            color='coral', alpha=0.8)
    
    ax4.set_ylabel('Score')
    ax4.set_title('Average Performance Metrics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}/comparison_plots.png")


def run_q3_experiments(save_dir='./results/q3'):
    """Run Q3 comparison experiments."""
    return comprehensive_comparison(save_dir)


if __name__ == "__main__":
    results = run_q3_experiments()
