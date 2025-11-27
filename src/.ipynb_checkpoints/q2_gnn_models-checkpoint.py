"""
q2_gnn_models.py - Q2: Graph Neural Networks for Graph Classification

This module implements:
1. Multiple GNN architectures (GCN, GIN, GraphSAGE, GAT)
2. Training and evaluation pipelines
3. Ablation studies on model parameters
4. Imbalanced data handling (class weights, focal loss, resampling)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, SAGEConv, GATConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import time
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Import local modules
from data_utils import load_mutag_dataset, split_dataset, get_data_loaders, Timer


# Set device (supports CUDA, MPS for Apple Silicon, or CPU)
def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

device = get_device()


# =============================================================================
# Cross-Validation Functions
# =============================================================================

def cross_validate_gnn(dataset, model_name, n_folds=5, epochs=100, lr=0.001,
                       hidden_dim=64, num_layers=3, loss_type='ce', 
                       class_weights=None, random_state=42, verbose=True):
    """
    Perform k-fold cross-validation for GNN models.
    
    Args:
        dataset: PyTorch Geometric dataset
        model_name: Name of GNN model ('GCN', 'GIN', 'GraphSAGE', 'GAT')
        n_folds: Number of CV folds
        epochs: Training epochs per fold
        lr: Learning rate
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        loss_type: Loss function type
        class_weights: Class weights for imbalanced data
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with CV results and statistics
    """
    labels = [data.y.item() for data in dataset]
    indices = np.arange(len(dataset))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_results = []
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation: {model_name} ({n_folds} folds)")
    print(f"{'='*60}")
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels)):
        if verbose:
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Further split train_val into train and validation
        train_val_labels = [labels[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=0.15, random_state=random_state,
            stratify=train_val_labels
        )
        
        # Create data loaders
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = get_model(
            model_name,
            num_features=dataset.num_node_features,
            hidden_dim=hidden_dim,
            num_classes=dataset.num_classes,
            num_layers=num_layers
        )
        
        # Train model
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, verbose=False,
            loss_type=loss_type, class_weights=class_weights
        )
        
        # Evaluate on test fold
        metrics, preds, labels_fold, probs = evaluate(model, test_loader)
        
        fold_results.append(metrics)
        all_preds.extend(preds)
        all_labels.extend(labels_fold)
        all_probs.extend(probs)
        
        if verbose:
            print(f"  Accuracy: {metrics['accuracy']:.4f}, "
                  f"Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                  f"F1: {metrics['f1']:.4f}")
    
    # Compute aggregate statistics
    results_df = pd.DataFrame(fold_results)
    
    cv_summary = {
        'model': model_name,
        'n_folds': n_folds,
        'metrics': {}
    }
    
    for col in results_df.columns:
        cv_summary['metrics'][col] = {
            'mean': results_df[col].mean(),
            'std': results_df[col].std(),
            'min': results_df[col].min(),
            'max': results_df[col].max()
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary: {model_name}")
    print(f"{'='*60}")
    
    key_metrics = ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'auc', 'mcc']
    for metric in key_metrics:
        if metric in cv_summary['metrics']:
            m = cv_summary['metrics'][metric]
            print(f"  {metric:20s}: {m['mean']:.4f} ± {m['std']:.4f} "
                  f"[{m['min']:.4f}, {m['max']:.4f}]")
    
    return {
        'fold_results': results_df,
        'summary': cv_summary,
        'all_predictions': np.array(all_preds),
        'all_labels': np.array(all_labels),
        'all_probs': np.array(all_probs)
    }


def cross_validate_all_models(dataset, n_folds=5, epochs=100, 
                               models=['GCN', 'GIN', 'GraphSAGE', 'GAT'],
                               save_dir='./results/cv'):
    """
    Run cross-validation for all GNN models.
    
    Args:
        dataset: PyTorch Geometric dataset
        n_folds: Number of CV folds
        epochs: Training epochs
        models: List of model names to evaluate
        save_dir: Directory to save results
        
    Returns:
        DataFrame with all CV results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = []
    
    for model_name in models:
        cv_results = cross_validate_gnn(
            dataset, model_name, n_folds=n_folds, epochs=epochs
        )
        
        # Extract summary for comparison
        summary = cv_results['summary']
        result_row = {'model': model_name}
        
        for metric, stats in summary['metrics'].items():
            result_row[f'{metric}_mean'] = stats['mean']
            result_row[f'{metric}_std'] = stats['std']
        
        all_results.append(result_row)
        
        # Save fold results
        cv_results['fold_results'].to_csv(
            f'{save_dir}/{model_name}_cv_folds.csv', index=False
        )
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{save_dir}/cv_comparison.csv', index=False)
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION COMPARISON (ALL MODELS)")
    print(f"{'='*70}")
    
    display_cols = ['model', 'accuracy_mean', 'accuracy_std', 
                    'balanced_accuracy_mean', 'f1_mean', 'auc_mean']
    available_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[available_cols].to_string(index=False))
    
    return results_df


# =============================================================================
# Loss Functions for Imbalanced Data
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in GNNs.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss using effective number of samples.
    """
    
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.tensor(weights).float()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)[targets]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = weights * focal_weight * ce_loss
        return loss.mean()


def get_class_weights(labels):
    """Compute class weights for imbalanced data."""
    labels = np.array(labels)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights).float()


def oversample_graph_dataset(dataset, indices):
    """
    Oversample minority class graphs by duplication.
    
    Args:
        dataset: PyTorch Geometric dataset
        indices: Training indices
        
    Returns:
        List of oversampled graph Data objects
    """
    labels = [dataset[i].y.item() for i in indices]
    class_counts = Counter(labels)
    majority_count = max(class_counts.values())
    
    print(f"Original distribution: {class_counts}")
    
    oversampled_data = []
    
    # Add all original graphs
    for idx in indices:
        oversampled_data.append(dataset[idx])
    
    # Oversample minority classes
    np.random.seed(42)
    for cls in class_counts:
        if class_counts[cls] < majority_count:
            cls_indices = [indices[i] for i, l in enumerate(labels) if l == cls]
            n_to_add = majority_count - class_counts[cls]
            
            selected = np.random.choice(cls_indices, size=n_to_add, replace=True)
            for idx in selected:
                original = dataset[idx]
                duplicate = Data(
                    x=original.x.clone(),
                    edge_index=original.edge_index.clone(),
                    y=original.y.clone()
                )
                if hasattr(original, 'edge_attr') and original.edge_attr is not None:
                    duplicate.edge_attr = original.edge_attr.clone()
                oversampled_data.append(duplicate)
    
    new_labels = [d.y.item() for d in oversampled_data]
    print(f"After oversampling: {Counter(new_labels)}")
    
    return oversampled_data


# =============================================================================
# GNN Model Architectures
# =============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network for graph classification."""
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2, 
                 num_layers=3, dropout=0.5, pooling='mean'):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network for graph classification."""
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2,
                 num_layers=3, dropout=0.5, pooling='add'):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = num_features if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier with jumping knowledge
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # Collect representations from all layers
        layer_outputs = []
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Global pooling for each layer
            if self.pooling == 'add':
                layer_outputs.append(global_add_pool(x, batch))
            elif self.pooling == 'mean':
                layer_outputs.append(global_mean_pool(x, batch))
            elif self.pooling == 'max':
                layer_outputs.append(global_max_pool(x, batch))
        
        # Concatenate all layer outputs (jumping knowledge)
        x = torch.cat(layer_outputs, dim=1)
        
        # Classification
        x = self.classifier(x)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE for graph classification."""
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2,
                 num_layers=3, dropout=0.5, pooling='mean'):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # SAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # SAGE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


class GAT(nn.Module):
    """Graph Attention Network for graph classification."""
    
    def __init__(self, num_features, hidden_dim=64, num_classes=2,
                 num_layers=3, dropout=0.5, pooling='mean', num_heads=4):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim // num_heads, 
                                  heads=num_heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                      heads=num_heads, dropout=dropout))
        
        # Final GAT layer (single head)
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, 
                                  concat=False, dropout=dropout))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, edge_index, batch):
        # GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def train_epoch(model, loader, optimizer, criterion):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total


def evaluate(model, loader):
    """Evaluate model on a data loader with imbalance-aware metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'mcc': matthews_corrcoef(all_labels, all_preds),
    }
    
    try:
        metrics['auc'] = roc_auc_score(all_labels, all_probs)
    except:
        metrics['auc'] = np.nan
    
    return metrics, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, 
                weight_decay=5e-4, patience=20, verbose=True,
                loss_type='ce', class_weights=None, focal_gamma=2.0):
    """
    Train a GNN model with early stopping and imbalance handling.
    
    Args:
        model: GNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        verbose: Print progress
        loss_type: Loss function type ('ce', 'weighted_ce', 'focal', 'class_balanced')
        class_weights: Class weights for weighted cross-entropy
        focal_gamma: Gamma parameter for focal loss
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Setup loss function based on imbalance handling strategy
    if loss_type == 'weighted_ce' and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if verbose:
            print(f"Using weighted cross-entropy loss with weights: {class_weights.cpu().numpy()}")
    elif loss_type == 'focal':
        alpha = class_weights.numpy().tolist() if class_weights is not None else None
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
        if verbose:
            print(f"Using focal loss with gamma={focal_gamma}")
    elif loss_type == 'class_balanced':
        # Get samples per class from training data
        train_labels = []
        for data in train_loader.dataset:
            train_labels.append(data.y.item())
        class_counts = Counter(train_labels)
        samples_per_class = [class_counts[i] for i in sorted(class_counts.keys())]
        criterion = ClassBalancedLoss(samples_per_class, gamma=focal_gamma)
        if verbose:
            print(f"Using class-balanced loss with samples: {samples_per_class}")
    else:
        criterion = nn.CrossEntropyLoss()
        if verbose and loss_type != 'ce':
            print("Using standard cross-entropy loss")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_balanced_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validation
        val_metrics, _, _, _ = evaluate(model, val_loader)
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1']
        val_balanced_acc = val_metrics['balanced_accuracy']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_balanced_acc'].append(val_balanced_acc)
        
        # Early stopping based on balanced accuracy for imbalanced data
        if val_balanced_acc > best_val_acc:
            best_val_acc = val_balanced_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, "
                  f"Val Balanced Acc={val_balanced_acc:.4f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    train_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    history['train_time'] = train_time
    history['best_epoch'] = len(history['train_loss']) - patience_counter
    
    return model, history


def get_model(model_name, num_features, hidden_dim=64, num_classes=2,
              num_layers=3, dropout=0.5, pooling='mean', **kwargs):
    """
    Factory function to create GNN models.
    
    Args:
        model_name: One of 'GCN', 'GIN', 'GraphSAGE', 'GAT'
        num_features: Number of input features
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        num_layers: Number of GNN layers
        dropout: Dropout rate
        pooling: Pooling method ('mean', 'max', 'add')
        
    Returns:
        GNN model instance
    """
    model_map = {
        'GCN': GCN,
        'GIN': GIN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    
    model_class = model_map[model_name]
    
    return model_class(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
        **kwargs
    )


# =============================================================================
# Ablation Studies
# =============================================================================

def ablation_gnn_layers(dataset, model_name, num_layers_list=[1, 2, 3, 4, 5],
                        hidden_dim=64, epochs=100):
    """
    Ablation study on number of GNN layers.
    
    Args:
        dataset: PyTorch Geometric dataset
        model_name: GNN model type
        num_layers_list: List of layer counts to test
        hidden_dim: Hidden dimension
        epochs: Maximum epochs
        
    Returns:
        DataFrame with results
    """
    results = []
    
    # Split dataset
    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size=32
    )
    
    for num_layers in num_layers_list:
        print(f"\n--- {model_name} with {num_layers} layers ---")
        
        model = get_model(
            model_name, 
            num_features=dataset.num_node_features,
            hidden_dim=hidden_dim,
            num_classes=dataset.num_classes,
            num_layers=num_layers
        )
        
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, verbose=False
        )
        
        # Test evaluation
        test_metrics, _, _, _ = evaluate(model, test_loader)
        
        # Inference time
        start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                _ = model(data.x, data.edge_index, data.batch)
        inference_time = time.time() - start_time
        
        results.append({
            'model': model_name,
            'num_layers': num_layers,
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'train_time': history['train_time'],
            'inference_time': inference_time,
            'best_epoch': history['best_epoch']
        })
    
    return pd.DataFrame(results)


def ablation_hidden_dim(dataset, model_name, hidden_dims=[32, 64, 128, 256],
                        num_layers=3, epochs=100):
    """
    Ablation study on hidden dimension.
    """
    results = []
    
    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size=32
    )
    
    for hidden_dim in hidden_dims:
        print(f"\n--- {model_name} with hidden_dim={hidden_dim} ---")
        
        model = get_model(
            model_name,
            num_features=dataset.num_node_features,
            hidden_dim=hidden_dim,
            num_classes=dataset.num_classes,
            num_layers=num_layers
        )
        
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, verbose=False
        )
        
        test_metrics, _, _, _ = evaluate(model, test_loader)
        
        results.append({
            'model': model_name,
            'hidden_dim': hidden_dim,
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'train_time': history['train_time'],
        })
    
    return pd.DataFrame(results)


def ablation_pooling(dataset, model_name, pooling_methods=['mean', 'max', 'add'],
                     num_layers=3, hidden_dim=64, epochs=100):
    """
    Ablation study on pooling methods.
    """
    results = []
    
    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size=32
    )
    
    for pooling in pooling_methods:
        print(f"\n--- {model_name} with pooling={pooling} ---")
        
        model = get_model(
            model_name,
            num_features=dataset.num_node_features,
            hidden_dim=hidden_dim,
            num_classes=dataset.num_classes,
            num_layers=num_layers,
            pooling=pooling
        )
        
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, verbose=False
        )
        
        test_metrics, _, _, _ = evaluate(model, test_loader)
        
        results.append({
            'model': model_name,
            'pooling': pooling,
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'train_time': history['train_time'],
        })
    
    return pd.DataFrame(results)


def ablation_learning_rate(dataset, model_name, learning_rates=[0.0001, 0.001, 0.01],
                           num_layers=3, hidden_dim=64, epochs=100):
    """
    Ablation study on learning rate.
    """
    results = []
    
    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size=32
    )
    
    for lr in learning_rates:
        print(f"\n--- {model_name} with lr={lr} ---")
        
        model = get_model(
            model_name,
            num_features=dataset.num_node_features,
            hidden_dim=hidden_dim,
            num_classes=dataset.num_classes,
            num_layers=num_layers
        )
        
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, lr=lr, verbose=False
        )
        
        test_metrics, _, _, _ = evaluate(model, test_loader)
        
        results.append({
            'model': model_name,
            'learning_rate': lr,
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'train_time': history['train_time'],
        })
    
    return pd.DataFrame(results)


def run_q2_experiments(save_dir='./results/q2'):
    """
    Run all Q2 experiments: GNN for Graph Classification.
    
    Args:
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 70)
    print("Q2: GRAPH NEURAL NETWORKS FOR GRAPH CLASSIFICATION")
    print("=" * 70)
    
    # Load dataset
    dataset, info = load_mutag_dataset()
    
    # Analyze class distribution
    labels = [data.y.item() for data in dataset]
    class_counts = Counter(labels)
    print(f"\nClass distribution: {class_counts}")
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Compute class weights
    class_weights = get_class_weights(labels)
    print(f"Class weights: {class_weights.numpy()}")
    
    # Split data
    train_idx, val_idx, test_idx = split_dataset(dataset)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx, batch_size=32
    )
    
    models_to_test = ['GCN', 'GIN', 'GraphSAGE', 'GAT']
    all_results = {}
    
    # =====================================================
    # Experiment 1: Train all GNN architectures (baseline)
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 1: Training All GNN Architectures (Baseline)")
    print("=" * 60)
    
    model_results = []
    trained_models = {}
    
    for model_name in models_to_test:
        print(f"\n--- Training {model_name} ---")
        
        model = get_model(
            model_name,
            num_features=dataset.num_node_features,
            hidden_dim=64,
            num_classes=dataset.num_classes,
            num_layers=3,
            dropout=0.5
        )
        
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=200, lr=0.001, verbose=True
        )
        
        # Test evaluation
        test_metrics, _, _, _ = evaluate(model, test_loader)
        
        # Inference time
        start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                _ = model(data.x, data.edge_index, data.batch)
        inference_time = time.time() - start_time
        
        test_metrics['train_time'] = history['train_time']
        test_metrics['inference_time'] = inference_time
        test_metrics['model'] = model_name
        
        model_results.append(test_metrics)
        trained_models[model_name] = model
        
        print(f"{model_name}: Acc={test_metrics['accuracy']:.4f}, "
              f"Balanced Acc={test_metrics['balanced_accuracy']:.4f}, "
              f"F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    model_results_df = pd.DataFrame(model_results)
    print("\n" + "=" * 40)
    print("GNN Model Comparison (Baseline):")
    print(model_results_df.to_string(index=False))
    model_results_df.to_csv(f'{save_dir}/gnn_comparison.csv', index=False)
    
    # =====================================================
    # Experiment 2: Imbalance Handling Strategies
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 2: Imbalance Handling Strategies")
    print("=" * 60)
    
    imbalance_results = []
    
    # Test different imbalance handling strategies with GIN (best performer usually)
    strategies = [
        ('baseline', 'ce', None),
        ('weighted_ce', 'weighted_ce', class_weights),
        ('focal_loss', 'focal', class_weights),
        ('class_balanced', 'class_balanced', None),
    ]
    
    for strategy_name, loss_type, weights in strategies:
        print(f"\n--- Testing {strategy_name} ---")
        
        model = get_model(
            'GIN',
            num_features=dataset.num_node_features,
            hidden_dim=64,
            num_classes=dataset.num_classes,
            num_layers=3
        )
        
        model, history = train_model(
            model, train_loader, val_loader,
            epochs=200, lr=0.001, verbose=False,
            loss_type=loss_type, class_weights=weights
        )
        
        test_metrics, _, _, _ = evaluate(model, test_loader)
        test_metrics['strategy'] = strategy_name
        test_metrics['train_time'] = history['train_time']
        imbalance_results.append(test_metrics)
        
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"  MCC: {test_metrics['mcc']:.4f}")
    
    # Test with oversampling
    print("\n--- Testing Graph Oversampling ---")
    oversampled_data = oversample_graph_dataset(dataset, train_idx)
    oversampled_loader = DataLoader(oversampled_data, batch_size=32, shuffle=True)
    
    model = get_model(
        'GIN',
        num_features=dataset.num_node_features,
        hidden_dim=64,
        num_classes=dataset.num_classes,
        num_layers=3
    )
    
    model, history = train_model(
        model, oversampled_loader, val_loader,
        epochs=200, lr=0.001, verbose=False
    )
    
    test_metrics, _, _, _ = evaluate(model, test_loader)
    test_metrics['strategy'] = 'oversampling'
    test_metrics['train_time'] = history['train_time']
    imbalance_results.append(test_metrics)
    
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  MCC: {test_metrics['mcc']:.4f}")
    
    imbalance_df = pd.DataFrame(imbalance_results)
    print("\n" + "=" * 40)
    print("Imbalance Handling Comparison (GIN):")
    cols = ['strategy', 'accuracy', 'balanced_accuracy', 'f1_macro', 'mcc', 'auc']
    print(imbalance_df[cols].to_string(index=False))
    imbalance_df.to_csv(f'{save_dir}/imbalance_handling.csv', index=False)
    
    # =====================================================
    # Experiment 3: Ablation on Number of Layers
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 3: Number of Layers Ablation")
    print("=" * 60)
    
    layers_results = []
    for model_name in ['GCN', 'GIN']:
        df = ablation_gnn_layers(dataset, model_name, 
                                 num_layers_list=[1, 2, 3, 4, 5])
        layers_results.append(df)
    
    layers_df = pd.concat(layers_results)
    print("\nLayers Ablation Results:")
    print(layers_df.to_string(index=False))
    layers_df.to_csv(f'{save_dir}/layers_ablation.csv', index=False)
    
    # =====================================================
    # Experiment 4: Ablation on Hidden Dimension
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 4: Hidden Dimension Ablation")
    print("=" * 60)
    
    hidden_results = []
    for model_name in ['GCN', 'GIN']:
        df = ablation_hidden_dim(dataset, model_name,
                                 hidden_dims=[32, 64, 128])
        hidden_results.append(df)
    
    hidden_df = pd.concat(hidden_results)
    print("\nHidden Dimension Ablation Results:")
    print(hidden_df.to_string(index=False))
    hidden_df.to_csv(f'{save_dir}/hidden_dim_ablation.csv', index=False)
    
    # =====================================================
    # Experiment 5: Ablation on Pooling Method
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 5: Pooling Method Ablation")
    print("=" * 60)
    
    pooling_results = []
    for model_name in ['GCN', 'GIN']:
        df = ablation_pooling(dataset, model_name,
                              pooling_methods=['mean', 'max', 'add'])
        pooling_results.append(df)
    
    pooling_df = pd.concat(pooling_results)
    print("\nPooling Method Ablation Results:")
    print(pooling_df.to_string(index=False))
    pooling_df.to_csv(f'{save_dir}/pooling_ablation.csv', index=False)
    
    # =====================================================
    # Experiment 6: Ablation on Learning Rate
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 6: Learning Rate Ablation")
    print("=" * 60)
    
    lr_results = []
    for model_name in ['GCN', 'GIN']:
        df = ablation_learning_rate(dataset, model_name,
                                    learning_rates=[0.0001, 0.001, 0.01])
        lr_results.append(df)
    
    lr_df = pd.concat(lr_results)
    print("\nLearning Rate Ablation Results:")
    print(lr_df.to_string(index=False))
    lr_df.to_csv(f'{save_dir}/learning_rate_ablation.csv', index=False)
    
    # =====================================================
    # Experiment 7: Cross-Validation
    # =====================================================
    print("\n" + "=" * 60)
    print("Experiment 7: 5-Fold Cross-Validation")
    print("=" * 60)
    
    cv_results = cross_validate_all_models(
        dataset, n_folds=5, epochs=100,
        models=['GCN', 'GIN', 'GraphSAGE', 'GAT'],
        save_dir=f'{save_dir}/cv'
    )
    
    # Save summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - Q2")
    print("=" * 60)
    print("\nBaseline Model Comparison:")
    print(model_results_df.to_string(index=False))
    print("\nBest Imbalance Handling Strategy:")
    best_strategy = imbalance_df.loc[imbalance_df['balanced_accuracy'].idxmax()]
    print(f"  Strategy: {best_strategy['strategy']}")
    print(f"  Balanced Accuracy: {best_strategy['balanced_accuracy']:.4f}")
    
    print("\nCross-Validation Results (5-fold):")
    cv_display_cols = ['model', 'accuracy_mean', 'accuracy_std', 'balanced_accuracy_mean']
    available_cols = [c for c in cv_display_cols if c in cv_results.columns]
    print(cv_results[available_cols].to_string(index=False))
    
    return {
        'model_results': model_results_df,
        'imbalance_results': imbalance_df,
        'layers_ablation': layers_df,
        'hidden_ablation': hidden_df,
        'pooling_ablation': pooling_df,
        'lr_ablation': lr_df,
        'cv_results': cv_results,
        'trained_models': trained_models
    }


if __name__ == "__main__":
    results = run_q2_experiments()
