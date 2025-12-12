# Graph Classification Project - MUTAG Dataset

## Project Overview

This project implements supervised machine learning on large-scale graph data using the MUTAG dataset. It includes multiple graph classification frameworks comparing classic machine learning with Graph Neural Networks.

**Key Features:**
- **MPS Support**: Optimized for Apple Silicon M4 MacBooks
- **Graph Mining** Subgraph Mining Pipeline implemented 
- **Cross-Validation**: 5-fold stratified cross-validation for reliable evaluation
- **Imbalanced Data Handling**: Multiple strategies for handling class imbalance
- **Comprehensive Metrics**: Balanced accuracy, MCC, macro-F1, and more

## Hardware Support

The project automatically detects and uses the best available hardware:
1. **CUDA** (NVIDIA GPUs)
2. **MPS** (Apple Silicon - M4 pro MacBooks)
3. **CPU** (fallback)

For our case, we run the experiment on M4 Pro MacBooks.

```python
# Device is automatically selected
from q2_gnn_models import device
print(f"Using: {device}")  # e.g., "mps" on MacBook Pro
```

## Dataset

**MUTAG Dataset**: A standard benchmark for graph classification containing 188 small molecular graphs representing chemical compounds. Each graph is labeled as mutagenic (1) or non-mutagenic (0).

- **Number of graphs**: 188
- **Average nodes**: 17.93 $\pm$ 4.58 per graph
- **Average edges**: 39.59 $\pm$ 11.37 per graph
- **Task**: Binary classification (mutagenic vs non-mutagenic)

## Project Structure

```
graph_project/
├── src/
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── q1_gspan_classic_ml.py # Q1: gSpan + Classic ML
│   ├── q2_gnn_models.py       # Q2: GNN architectures
│   ├── q3_comparison.py       # Q3: Comparison analysis
│   ├── q4_explainability.py   # Q4: Explainability
│   └── main.py                # Main runner script
├── results/                   # Output directory for results
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For PyTorch Geometric, you may need to install with specific CUDA version:
```bash
# CPU only
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

# With CUDA 11.8
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Running Experiments

### Run All Experiments
```bash
cd src
python main.py
```

### Run Individual Components
```bash
# Q1: Frequent Subgraph Mining + Classic ML
python q1_gspan_classic_ml.py

# Q2: Graph Neural Networks
python q2_gnn_models.py

# Q3: Comparison
python q3_comparison.py

# Q4: Explainability
python q4_explainability.py
```

### Cross-Validation Examples
```python
from data_utils import load_mutag_dataset
from q2_gnn_models import cross_validate_gnn, cross_validate_all_models

# Load dataset
dataset, info = load_mutag_dataset()

# Cross-validate a single model
cv_results = cross_validate_gnn(
    dataset, 
    model_name='GIN',
    n_folds=5,
    epochs=100,
    loss_type='focal'  # Use focal loss for imbalanced data
)
print(f"Accuracy: {cv_results['summary']['metrics']['accuracy']['mean']:.4f} "
      f"± {cv_results['summary']['metrics']['accuracy']['std']:.4f}")

# Cross-validate all models
all_cv_results = cross_validate_all_models(
    dataset, 
    n_folds=5, 
    models=['GCN', 'GIN', 'GraphSAGE', 'GAT']
)
```

### Classic ML Cross-Validation
```python
from q1_gspan_classic_ml import cross_validate_all_classic_ml

# Assuming X and y are your feature matrix and labels
cv_results = cross_validate_all_classic_ml(X, y, n_folds=5)
```

## Component Details

### Q1: Frequent Subgraph Mining + Classic ML

**Approach**:
1. Mine frequent subgraphs using gSpan-style algorithm
2. Construct feature vectors based on subgraph counts
3. Train classic ML models: Random Forest, SVM, Gradient Boosting

**Ablation Studies**:
- Mining support threshold: [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
- Random Forest: n_estimators, max_depth
- SVM: C, kernel type
- Gradient Boosting: n_estimators, learning_rate

**Imbalance Handling**:
- Class weighting (balanced weights)
- SMOTE oversampling
- Random undersampling


### Q2: Graph Neural Networks

**Architectures**:
- **GCN**: Graph Convolutional Network
- **GIN**: Graph Isomorphism Network
- **GraphSAGE**: Sample and Aggregate
- **GAT**: Graph Attention Network

**Ablation Studies**:
- Number of layers: [1, 2, 3, 4, 5]
- Hidden dimensions: [32, 64, 128]
- Pooling methods: [mean, max, add]
- Learning rates: [0.0001, 0.001, 0.01]

**Imbalance Handling**:
- Weighted cross-entropy loss
- Focal Loss (gamma=2.0)
- Class-Balanced Loss (effective number of samples)
- Graph-level oversampling (duplication)

**Cross-Validation**:
- 5-fold stratified cross-validation
- Reports mean ± std for all metrics
- Comparison across all GNN architectures

### Q3: Comparison

**Metrics Compared**:
- Accuracy
- **Balanced Accuracy** (important for imbalanced data)
- Precision (weighted and macro)
- Recall (weighted and macro)
- F1-Score (weighted and macro)
- **Matthews Correlation Coefficient (MCC)**
- AUC-ROC
- Training time
- Inference time

**Visualizations**:
- Accuracy comparison bar charts
- F1 score comparison
- Training time comparison
- Performance metrics radar chart

### Q4: Explainability

**GNNExplainer**:
- Post-hoc explanation method for GNNs
- Identifies important nodes and edges for predictions

**Evaluation Metrics**:
- **Fidelity+**: Faithfulness when keeping important features
- **Fidelity-**: Change in prediction when removing important features
- **Sparsity**: Compactness of explanations
- **Runtime**: Computation time per explanation

**Classic ML Interpretability**:
- Feature importance from Random Forest
- Coefficient analysis for SVM
- Semantic meaning from frequent subgraph patterns

## Output Files

### Q1 Results (./results/q1/)
- `threshold_ablation.csv`: Support threshold ablation results
- `random_forest_ablation.csv`: RF parameter ablation
- `svm_ablation.csv`: SVM parameter ablation
- `gradient_boosting_ablation.csv`: GB parameter ablation
- `rf_feature_importance.csv`: Feature importance scores
- `q1_summary.csv`: Overall summary

### Q2 Results (./results/q2/)
- `gnn_comparison.csv`: All GNN models comparison
- `layers_ablation.csv`: Number of layers ablation
- `hidden_dim_ablation.csv`: Hidden dimension ablation
- `pooling_ablation.csv`: Pooling method ablation
- `learning_rate_ablation.csv`: Learning rate ablation

### Q3 Results (./results/q3/)
- `comparison_results.csv`: Full comparison table
- `timing_breakdown.csv`: Timing analysis
- `summary_statistics.csv`: Statistical summary
- `comparison_plots.png`: Visualization

### Q4 Results (./results/q4/)
- `gnnexplainer_results.csv`: Explanation metrics per graph
- `classic_ml_feature_importance.csv`: Feature importance
- `explainability_summary.csv`: Summary statistics
- `comparison_summary.txt`: Detailed comparison
- `explainability_plots.png`: Visualization

## Key Findings

1. **Classic ML with gSpan**:
   - Provides interpretable features (frequent subgraphs)
   - Fast inference after feature extraction
   - Good performance with proper threshold tuning

2. **GNN Models**:
   - End-to-end learning without feature engineering
   - GIN typically achieves best performance
   - More computationally expensive during training

3. **Comparison**:
   - GNNs generally achieve higher accuracy
   - Classic ML provides faster inference
   - Both approaches are competitive on MUTAG

4. **Explainability**:
   - GNNExplainer provides local explanations
   - Classic ML provides global, interpretable features
   - Trade-off between expressiveness and interpretability

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- scikit-learn 1.3+
- numpy, pandas, matplotlib, seaborn
- networkx 3.1+

## References

- MUTAG Dataset: Debnath et al. (1991)
- GCN: Kipf & Welling (2017)
- GIN: Xu et al. (2019)
- GraphSAGE: Hamilton et al. (2017)
- GAT: Veličković et al. (2018)
- GNNExplainer: Ying et al. (2019)
- gSpan: Yan & Han (2002)

## Authors

- Alireza Hoseinpour
- Emmanuel Agbeli
- Rajni Hiroshima
- Rahubadde De Silva
- Cadence Litteral