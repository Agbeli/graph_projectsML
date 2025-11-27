"""
main.py - Main Runner for Graph Classification Project

This script runs all experiments for Q1-Q4:
- Q1: Frequent Subgraph Mining + Classic ML
- Q2: Graph Neural Networks
- Q3: Comparison of approaches
- Q4: Explainability analysis
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import load_mutag_dataset
from q1_gspan_classic_ml import run_q1_experiments
from q2_gnn_models import run_q2_experiments
from q3_comparison import run_q3_experiments
from q4_explainability import run_q4_experiments


def print_header(title):
    """Print a formatted section header."""
    print("\n")
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main():
    """Run all experiments."""
    
    start_time = time.time()
    
    print_header("GRAPH CLASSIFICATION PROJECT - MUTAG DATASET")
    print("\nThis project implements:")
    print("  Q1: Frequent Subgraph Mining + Classic ML (Random Forest, SVM)")
    print("  Q2: Graph Neural Networks (GCN, GIN, GraphSAGE, GAT)")
    print("  Q3: Comprehensive comparison of both approaches")
    print("  Q4: Explainability analysis (GNNExplainer vs Classic ML)")
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # =========================================================================
    # Q1: Frequent Subgraph Mining + Classic ML
    # =========================================================================
    print_header("Q1: FREQUENT SUBGRAPH MINING + CLASSIC ML")
    
    try:
        q1_results = run_q1_experiments(save_dir='./results/q1')
        all_results['q1'] = q1_results
        print("\n✓ Q1 experiments completed successfully!")
    except Exception as e:
        print(f"\n✗ Q1 experiments failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Q2: Graph Neural Networks
    # =========================================================================
    print_header("Q2: GRAPH NEURAL NETWORKS")
    
    try:
        q2_results = run_q2_experiments(save_dir='./results/q2')
        all_results['q2'] = q2_results
        print("\n✓ Q2 experiments completed successfully!")
    except Exception as e:
        print(f"\n✗ Q2 experiments failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Q3: Comparison
    # =========================================================================
    print_header("Q3: COMPREHENSIVE COMPARISON")
    
    try:
        q3_results = run_q3_experiments(save_dir='./results/q3')
        all_results['q3'] = q3_results
        print("\n✓ Q3 experiments completed successfully!")
    except Exception as e:
        print(f"\n✗ Q3 experiments failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Q4: Explainability
    # =========================================================================
    print_header("Q4: EXPLAINABILITY ANALYSIS")
    
    try:
        q4_results = run_q4_experiments(save_dir='./results/q4')
        all_results['q4'] = q4_results
        print("\n✓ Q4 experiments completed successfully!")
    except Exception as e:
        print(f"\n✗ Q4 experiments failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    total_time = time.time() - start_time
    
    print_header("EXPERIMENT SUMMARY")
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("\nResults saved to:")
    print("  - ./results/q1/ (Classic ML with gSpan)")
    print("  - ./results/q2/ (GNN models)")
    print("  - ./results/q3/ (Comparison)")
    print("  - ./results/q4/ (Explainability)")
    
    print("\nKey files generated:")
    for q in ['q1', 'q2', 'q3', 'q4']:
        result_dir = f'./results/{q}'
        if os.path.exists(result_dir):
            files = os.listdir(result_dir)
            print(f"  {q}: {', '.join(files)}")
    
    return all_results


if __name__ == "__main__":
    results = main()
