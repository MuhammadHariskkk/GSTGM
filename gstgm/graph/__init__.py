"""Scene-centric graphs: kernels, adjacency, construction, and message passing (Phase 3)."""

from gstgm.graph.adjacency import (
    build_normalized_adjacency,
    build_weight_matrix,
    masked_outer_node_mask,
    symmetric_normalized_adjacency,
)
from gstgm.graph.graph_builder import (
    SceneGraphBatch,
    build_from_collated_batch,
    build_scene_graph_batch,
    graph_config,
    stack_scene_nodes,
    stacked_node_velocities,
)
from gstgm.graph.kernels import (
    apply_similarity_kernel,
    gstgm_adjacency_similarity,
    inverse_sq_euclidean_weights,
    pairwise_squared_euclidean,
)
from gstgm.graph.message_passing import GraphConv, batched_adjacency_aggregate

__all__ = [
    "GraphConv",
    "SceneGraphBatch",
    "apply_similarity_kernel",
    "batched_adjacency_aggregate",
    "build_from_collated_batch",
    "build_normalized_adjacency",
    "build_scene_graph_batch",
    "build_weight_matrix",
    "graph_config",
    "gstgm_adjacency_similarity",
    "inverse_sq_euclidean_weights",
    "masked_outer_node_mask",
    "pairwise_squared_euclidean",
    "stack_scene_nodes",
    "stacked_node_velocities",
    "symmetric_normalized_adjacency",
]
