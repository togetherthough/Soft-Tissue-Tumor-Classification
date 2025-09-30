"""
GeoTopo-STS: Geometry-Topology Soft Tissue Sarcoma Classification
==================================================================

A multi-modal deep learning pipeline combining:
- Voxel pathway: 3D CNN/Mamba with ROI-aware pooling
- Geometry pathway: Surface mesh GNN with curvature & intensity features
- Topology pathway: Persistent homology features
- Hierarchical fusion and classification
"""

__version__ = "0.1.0"

from .models import *
from .dataio import *
from .geometry import *
from .topology import *
