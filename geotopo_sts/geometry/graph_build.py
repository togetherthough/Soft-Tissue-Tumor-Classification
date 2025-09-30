"""
Graph construction from meshes and skeletons
Creates edge indices and edge attributes for GNN processing
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial import cKDTree


def build_mesh_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    k_neighbors: int = 8,
    max_edge_length_mm: Optional[float] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Build graph from mesh for GNN processing.
    
    Edges come from:
    1. Face connectivity (mesh topology)
    2. k-nearest neighbors in 3D space
    
    Args:
        vertices: (N, 3) vertex positions in mm
        faces: (M, 3) triangle faces
        k_neighbors: number of nearest neighbors to connect
        max_edge_length_mm: optional max edge length to prune distant edges
    
    Returns:
        edge_index: (2, E) array of edge pairs
        edge_attr: (E, D) edge attributes (relative vector, distance)
    """
    if len(vertices) == 0:
        return np.zeros((2, 0), dtype=np.int64), None
    
    edges_set = set()
    
    # Add edges from faces
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edges_set.add((min(v1, v2), max(v1, v2)))
    
    # Add k-NN edges
    if k_neighbors > 0:
        tree = cKDTree(vertices)
        distances, indices = tree.query(vertices, k=k_neighbors + 1)  # +1 because includes self
        
        for i in range(len(vertices)):
            for j, d in zip(indices[i, 1:], distances[i, 1:]):  # skip self
                if max_edge_length_mm is None or d <= max_edge_length_mm:
                    edges_set.add((min(i, j), max(i, j)))
    
    # Convert to array
    if len(edges_set) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edges = np.array(list(edges_set), dtype=np.int64)
        # Make undirected: add both (i,j) and (j,i)
        edge_index = np.concatenate([edges, edges[:, [1, 0]]], axis=0).T
    
    # Compute edge attributes
    if len(edge_index) > 0:
        src, dst = edge_index
        edge_vectors = vertices[dst] - vertices[src]  # (E, 3)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)  # (E, 1)
        
        edge_attr = np.concatenate([edge_vectors, edge_lengths], axis=1).astype(np.float32)
    else:
        edge_attr = None
    
    return edge_index, edge_attr


def build_skeleton_graph(
    nodes: np.ndarray,
    adjacency: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build graph from skeleton nodes and adjacency.
    
    Args:
        nodes: (N, 3) node positions in mm
        adjacency: (N, N) adjacency matrix or list of edges
    
    Returns:
        edge_index: (2, E)
        edge_attr: (E, D) with length, normalized direction
    """
    if isinstance(adjacency, list):
        # List of (i, j) pairs
        edges = np.array(adjacency, dtype=np.int64)
    else:
        # Adjacency matrix
        edges = np.argwhere(adjacency > 0)
    
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    
    # Make undirected
    edges_undirected = np.concatenate([edges, edges[:, [1, 0]]], axis=0)
    edge_index = edges_undirected.T
    
    # Edge attributes
    src, dst = edge_index
    edge_vectors = nodes[dst] - nodes[src]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1, keepdims=True)
    edge_directions = edge_vectors / (edge_lengths + 1e-8)
    
    edge_attr = np.concatenate([edge_directions, edge_lengths], axis=1).astype(np.float32)
    
    return edge_index, edge_attr
