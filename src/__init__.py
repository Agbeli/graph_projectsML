"""
Graph Classification Project - MUTAG Dataset

This package implements supervised machine learning on graph data:
- Q1: Frequent Subgraph Mining + Classic ML
- Q2: Graph Neural Networks (GCN, GIN, GraphSAGE, GAT)
- Q3: Comparison of approaches
- Q4: Explainability analysis
"""

from .data_utils import load_mutag_dataset, pyg_to_networkx_graphs, split_dataset
from .q1_gspan_classic_ml import run_q1_experiments
from .q2_gnn_models import run_q2_experiments
from .q3_comparison import run_q3_experiments
from .q4_explainability import run_q4_experiments

__version__ = "1.0.0"
__all__ = [
    'load_mutag_dataset',
    'pyg_to_networkx_graphs', 
    'split_dataset',
    'run_q1_experiments',
    'run_q2_experiments',
    'run_q3_experiments',
    'run_q4_experiments'
]
