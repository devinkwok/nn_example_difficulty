from typing import List
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

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


def supervised_prototypes(representations: torch.Tensor, labels: torch.Tensor, norm_order: int=2):
    """From
    Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. (2022).
    Beyond neural scaling laws: beating power law scaling via data pruning.
    Advances in Neural Information Processing Systems, 35, 19523-19536.

    Args:
        representations (torch.Tensor): per-example activations. In Sorscher et al. (2022),
            this is the embedding space of an ImageNet pre-trained self-supervised model SWaV.
            Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
        labels (torch.Tensor): true class labels of examples, with shape (N,)
        norm_order (int): type of norm to compute distance over. Defaults to L^2.

    Returns:
        torch.Tensor: distance from examples to label centroids (prototypes)
            in representation space, of shape (N,)
    """
    distances = torch.empty(representations.shape[0], dtype=representations.dtype, device=representations.device)
    for i in torch.unique(labels):
        # compute centroid as mean over representations of examples
        idx = (labels == i)
        embeddings = representations[idx, ...]
        prototype = torch.mean(embeddings, dim=0).broadcast_to(embeddings.shape)
        # for each example with this label, compute distance to correct label centroid
        distances[idx] = torch.linalg.vector_norm(embeddings - prototype, ord=norm_order)
    return distances


def self_supervised_prototypes(representations: torch.Tensor, k: int, max_iter: int=300):
    """From
    Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. (2022).
    Beyond neural scaling laws: beating power law scaling via data pruning.
    Advances in Neural Information Processing Systems, 35, 19523-19536.

    Args:
        representations (torch.Tensor): per-example activations. In Sorscher et al. (2022),
            this is the embedding space of an ImageNet pre-trained self-supervised model SWaV.
            Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
        k (int): number of means for k-means clustering.
            Sorscher et al. (2022) recommends any value within an order of magnitude of the true number classes,
            noting that the performance when using this metric for data pruning is not very sensitive to k.

    Returns:
        torch.Tensor: distance from examples to cluster centroids (self-supervised prototypes)
            in representation space, of shape (N,)
    """
    representations = representations.reshape(representations.shape[0], -1)
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    kmeans.fit(representations.detach().cpu().numpy())
    distances = kmeans.transform(representations)
    distances = torch.tensor(distances, dtype=representations.dtype, device=representations.device)
    distance_to_nearest_centroid = torch.min(distances, dim=-1).values
    return distance_to_nearest_centroid
