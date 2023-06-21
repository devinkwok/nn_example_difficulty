from typing import List
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from difficulty.metrics.forget import first_unforgettable


__all__ = [
    "prediction_depth",
    "supervised_prototypes",
    "self_supervised_prototypes",
]


def _nearest_neighbour(representations: torch.Tensor, query_points: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    knn = KNeighborsClassifier(k)  # use standard Euclidean distance
    representations = representations.reshape(representations.shape[0], -1).detach().cpu().numpy()
    query_points = query_points.reshape(query_points.shape[0], -1).detach().cpu().numpy()
    knn.fit(query_points, labels.detach().cpu().numpy())
    predictions = knn.predict(representations)
    return torch.tensor(predictions).to(dtype=labels.dtype, device=labels.device)


def prediction_depth(intermediate_activations: List[torch.Tensor], consensus_labels: torch.Tensor,
                     query_activations: List[torch.Tensor], query_labels: torch.Tensor, k: int=30) -> np.ndarray:
    """From
    Baldock, R., Maennel, H., and Neyshabur, B. (2021).
    Deep learning through the lens of example difficulty.
    Advances In Neural Information Processing Systems, 34.

    Args:
        intermediate_activations (List[torch.Tensor]): per-example activations in order of layer depth.
            Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened
        consensus_labels (torch.Tensor): labels to compare against KNN predictions, with shape (N,)
            In Baldock et al. (2021), these are set to the predictions of a majority of ensembled models.
        query_activations (List[torch.Tensor]): activations for examples used as KNN query points,
            with shape (M, ...). In Baldock et al. (2021), these are the entire training dataset.
        query_labels (torch.Tensor): labels to match to KNN predictions, with shape (M,)
            In Baldock et al. (2021), these are the consensus labels of the query examples.

    Returns:
        torch.Tensor: prediction depth (note: this is NOT backprop-enabled)
    """
    predictions = [_nearest_neighbour(v, q, query_labels, k)
                   for v, q in zip(intermediate_activations, query_activations)]
    # dims L*C where L is intermediate layers, C is classes
    layer_predict = torch.stack(predictions, dim=0)
    match = (layer_predict == consensus_labels.broadcast_to(layer_predict.shape))
    return first_unforgettable(match)


# TODO (self)-supervised prototypes sorscher
def supervised_prototypes(representations: np.ndarray, labels: np.ndarray):
    pass  #TODO distance to known label centroids


def self_supervised_prototypes(representations: np.ndarray, k: int):
    pass  #TODO distance after k-means clustering
