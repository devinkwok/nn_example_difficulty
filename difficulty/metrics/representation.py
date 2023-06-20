from typing import List
from collections import OrderedDict
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier

from difficulty.metrics.forget import first_unforgettable
from difficulty.model.eval import match_key


def _nearest_neighbour(representations: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    knn = KNeighborsClassifier(k)  # use standard Euclidean distance
    representations = representations.reshape(representations.shape[0], -1).detach().cpu().numpy()
    knn.fit(representations, labels.detach().cpu().numpy())
    predictions = knn.predict(representations)
    return torch.tensor(predictions).to(dtype=labels.dtype, device=labels.device)


def prediction_depth(intermediate_activations: OrderedDict, consensus_labels: torch.Tensor, k: int=30, include: List[str] = None, exclude: List[str] = None) -> np.ndarray:
    """From
    Baldock, R., Maennel, H., and Neyshabur, B. (2021).
    Deep learning through the lens of example difficulty.
    Advances In Neural Information Processing Systems, 34.

    Returns:
        torch.Tensor: prediction depth (note: this is NOT backprop-enabled)
    """
    predictions = []
    for name, layer in intermediate_activations.items():
        if match_key(name, include=include, exclude=exclude):
            predictions.append(_nearest_neighbour(layer, consensus_labels, k))
    # dims L*C where L is intermediate layers, C is classes
    layer_predict = torch.stack(predictions, dim=0)
    match = (layer_predict == consensus_labels.broadcast_to(layer_predict.shape))
    return first_unforgettable(match)


# TODO (self)-supervised prototypes sorscher
def supervised_prototypes(representations: np.ndarray, labels: np.ndarray):
    pass  #TODO distance to known label centroids


def self_supervised_prototypes(representations: np.ndarray, k: int):
    pass  #TODO distance after k-means clustering
