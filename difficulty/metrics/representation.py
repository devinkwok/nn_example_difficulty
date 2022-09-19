from typing import Dict, List
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from difficulty.metrics.forget import forgetting_events, first_unforgettable
from difficulty.model.eval import match_key


def _nearest_neighbour(representations: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    knn = KNeighborsClassifier(k)  # use standard Euclidean distance
    representations = representations.reshape(representations.shape[0], -1)
    knn.fit(representations, labels)
    return knn.predict(representations)


def prediction_depth(intermediate_activations: Dict[str, np.ndarray], consensus_labels: np.ndarray, k: int=30, include: List[str] = None, exclude: List[str] = None) -> np.ndarray:
    """From
    Baldock, R., Maennel, H., and Neyshabur, B. (2021).
    Deep learning through the lens of example difficulty.
    Advances In Neural Information Processing Systems, 34.

    Returns:
        np.ndarray: prediction depth
    """
    layers = intermediate_activations
    match_key(k, include=include, exclude=exclude)
    layer_predict = np.stack([_nearest_neighbour(x, consensus_labels, k) for x in layers.values()], axis=0)
    predict_output = forgetting_events(layer_predict == consensus_labels)
    return first_unforgettable(predict_output)


# TODO (self)-supervised prototypes sorscher
