"""
data_utils.py - Data loading and preprocessing utilities for MUTAG dataset
"""

import numpy as np
import networkx as nx
from collections import defaultdict
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import time


def load_mutag_dataset(root='./data'):
    """
    Load the MUTAG dataset using PyTorch Geometric.
    
    Returns:
        dataset: TUDataset object containing MUTAG graphs
        info: Dictionary with dataset statistics
    """
    dataset = TUDataset(root=root, name='MUTAG')
    
    # Compute dataset statistics
    num_graphs = len(dataset)
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    
    # Get class distribution
    labels = [data.y.item() for data in dataset]
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[label] += 1
    
    # Compute graph statistics
    num_nodes_list = [data.num_nodes for data in dataset]
    num_edges_list = [data.num_edges for data in dataset]
    
    info = {
        'num_graphs': num_graphs,
        'num_features': num_features,
        'num_classes': num_classes,
        'class_distribution': dict(class_counts),
        'avg_nodes': np.mean(num_nodes_list),
        'std_nodes': np.std(num_nodes_list),
        'avg_edges': np.mean(num_edges_list),
        'std_edges': np.std(num_edges_list),
        'min_nodes': min(num_nodes_list),
        'max_nodes': max(num_nodes_list),
        'min_edges': min(num_edges_list),
        'max_edges': max(num_edges_list),
    }
    
    print("=" * 60)
    print("MUTAG Dataset Statistics")
    print("=" * 60)
    print(f"Number of graphs: {info['num_graphs']}")
    print(f"Number of node features: {info['num_features']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class distribution: {info['class_distribution']}")
    print(f"Average nodes per graph: {info['avg_nodes']:.2f} ± {info['std_nodes']:.2f}")
    print(f"Average edges per graph: {info['avg_edges']:.2f} ± {info['std_edges']:.2f}")
    print(f"Node range: [{info['min_nodes']}, {info['max_nodes']}]")
    print(f"Edge range: [{info['min_edges']}, {info['max_edges']}]")
    print("=" * 60)
    
    return dataset, info


def pyg_to_networkx_graphs(dataset):
    """
    Convert PyTorch Geometric dataset to list of NetworkX graphs.
    
    Args:
        dataset: PyTorch Geometric TUDataset
        
    Returns:
        graphs: List of NetworkX graphs with node labels
        labels: List of graph labels
    """
    graphs = []
    labels = []
    
    for data in dataset:
        # Convert to NetworkX
        G = to_networkx(data, node_attrs=['x'], to_undirected=True)
        
        # Add node labels (using argmax of one-hot encoded features)
        for node in G.nodes():
            if G.nodes[node].get('x') is not None:
                # Get the node feature and convert to label
                x = G.nodes[node]['x']
                if hasattr(x, '__len__'):
                    G.nodes[node]['label'] = int(np.argmax(x))
                else:
                    G.nodes[node]['label'] = int(x)
            else:
                G.nodes[node]['label'] = 0
        
        graphs.append(G)
        labels.append(data.y.item())
    
    return graphs, labels


def convert_to_gspan_format(graphs, labels, output_file):
    """
    Convert NetworkX graphs to gSpan input format.
    
    gSpan format:
    t # N (transaction N)
    v node_id node_label
    e src_id dst_id edge_label
    
    Args:
        graphs: List of NetworkX graphs
        labels: List of graph labels
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        for idx, (G, label) in enumerate(zip(graphs, labels)):
            f.write(f"t # {idx}\n")
            
            # Write vertices
            node_mapping = {node: i for i, node in enumerate(G.nodes())}
            for node, new_id in node_mapping.items():
                node_label = G.nodes[node].get('label', 0)
                f.write(f"v {new_id} {node_label}\n")
            
            # Write edges (only once for undirected)
            seen_edges = set()
            for u, v in G.edges():
                if (v, u) not in seen_edges:
                    u_id, v_id = node_mapping[u], node_mapping[v]
                    edge_label = G.edges[u, v].get('label', 0)
                    f.write(f"e {u_id} {v_id} {edge_label}\n")
                    seen_edges.add((u, v))
            
            f.write("\n")
    
    print(f"Converted {len(graphs)} graphs to gSpan format: {output_file}")


def split_dataset(dataset, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: PyTorch Geometric dataset
        test_size: Fraction of data for testing
        val_size: Fraction of training data for validation
        random_state: Random seed
        
    Returns:
        train_idx, val_idx, test_idx: Indices for each split
    """
    labels = [data.y.item() for data in dataset]
    indices = np.arange(len(dataset))
    
    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Second split: train vs val
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=train_val_labels
    )
    
    print(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    return train_idx, val_idx, test_idx


def get_data_loaders(dataset, train_idx, val_idx, test_idx, batch_size=32):
    """
    Create PyTorch Geometric data loaders.
    
    Args:
        dataset: PyTorch Geometric dataset
        train_idx, val_idx, test_idx: Indices for each split
        batch_size: Batch size for training
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    train_dataset = dataset[train_idx.tolist()]
    val_dataset = dataset[val_idx.tolist()]
    test_dataset = dataset[test_idx.tolist()]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def get_cv_splits(dataset, n_splits=5, random_state=42):
    """
    Get cross-validation splits for the dataset.
    
    Args:
        dataset: PyTorch Geometric dataset
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    labels = [data.y.item() for data in dataset]
    indices = np.arange(len(dataset))
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, test_idx in skf.split(indices, labels):
        splits.append((train_idx, test_idx))
    
    return splits


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f} seconds")


if __name__ == "__main__":
    # Test data loading
    dataset, info = load_mutag_dataset()
    
    # Test conversion to NetworkX
    graphs, labels = pyg_to_networkx_graphs(dataset)
    print(f"\nConverted {len(graphs)} graphs to NetworkX format")
    
    # Test gSpan format conversion
    os.makedirs('./data/gspan', exist_ok=True)
    convert_to_gspan_format(graphs, labels, './data/gspan/mutag.txt')
    
    # Test dataset splitting
    train_idx, val_idx, test_idx = split_dataset(dataset)
    
    # Test data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset, train_idx, val_idx, test_idx
    )
    print(f"\nDataLoaders created: {len(train_loader)} train batches")
