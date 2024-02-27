from typing import List, Iterable, Tuple
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from difficulty.metrics.forget import first_unforgettable

from difficulty.utils import detach_tensors
from difficulty.model.eval import evaluate_intermediates, combine_batches


__all__ = [
    "representation_metrics",
    "PredictionDepth",
    "prediction_depth",
    "supervised_prototypes",
    "self_supervised_prototypes",
]


def representation_metrics(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device="cuda",
        to_cpu=True,
        to_numpy=False,
        named_modules: Iterable[Tuple[str, nn.Module]]=None,
        include: List[str]=None,
        exclude: List[str]=None,
        verbose=False,
        pd_k: int=30,
        pd_train_labels: torch.Tensor=None,
        pd_test_dataloader: torch.utils.data.DataLoader=None,
        proto_layer: str=None,
        selfproto_k: int=30,
        selfproto_random_state: int=None,
        selfproto_max_iter: int=300,
    ):
        generator = evaluate_intermediates(model, dataloader, device, named_modules, include, exclude, verbose)
        _, intermediates, _, labels = combine_batches(generator)

        # use true labels as consensus labels if not set
        train_labels = labels if pd_train_labels is None else pd_train_labels
        pd_obj = PredictionDepth(intermediates, train_labels, k=pd_k)
        # compute prediction depth on knn training data if test data not provided
        if pd_test_dataloader is None:
            pd = pd_obj(intermediates, train_labels)
        else:
            pd = []
            test_generator = evaluate_intermediates(model, pd_test_dataloader, device, named_modules, include, exclude, verbose)
            for _, test_intermediates, _, test_labels in test_generator:
                pd.append(pd_obj(test_intermediates, test_labels))
            pd = torch.cat(pd, dim=0)

        # use last layer of intermediates as representation by default
        if proto_layer is None:
            *_, proto_layer = intermediates.keys()
        representations = intermediates[proto_layer]
        proto = supervised_prototypes(representations, labels)
        selfproto = self_supervised_prototypes(representations, k=selfproto_k, max_iter=selfproto_max_iter, random_state=selfproto_random_state)

        return detach_tensors({
             "pd": pd,
             "proto": proto,
             "selfproto": selfproto,
        }, to_cpu=to_cpu, to_numpy=to_numpy)


def nearest_neighbour(representations: torch.Tensor, query_points: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    knn = KNeighborsClassifier(k)  # use standard Euclidean distance
    representations = representations.reshape(representations.shape[0], -1).detach().cpu().numpy()
    query_points = query_points.reshape(query_points.shape[0], -1).detach().cpu().numpy()
    knn.fit(query_points, labels.detach().cpu().numpy())
    predictions = knn.predict(representations)
    return torch.tensor(predictions).to(dtype=labels.dtype, device=labels.device)


class PredictionDepth:
    def __init__(self,
            intermediate_activations: Iterable,
            consensus_labels: torch.Tensor,
            k: int=30
    ) -> None:
        """From
        Baldock, R., Maennel, H., and Neyshabur, B. (2021).
        Deep learning through the lens of example difficulty.
        Advances In Neural Information Processing Systems, 34.

        Args:
            intermediate_activations (Iterable): list or ordered dict of activations by layer for examples used to fit the KNN,
                with shape (M, ...). In Baldock et al. (2021), these are the entire training dataset.
            consensus_labels (torch.Tensor): labels used to fit the KNN, with shape (M,)
                In Baldock et al. (2021), these are the consensus labels of the knn_activations.
            k (int, optional): number of neighbours to compare in k-nearest neighbours. Defaults to 30.
        """
        labels = consensus_labels.detach().cpu().numpy()
        self.knns = []
        for intermediates in self._dict_to_list(intermediate_activations):
            knn = KNeighborsClassifier(k)  # use standard Euclidean distance
            intermediates = intermediates.reshape(intermediates.shape[0], -1).detach().cpu().numpy()
            knn.fit(intermediates, labels)
            self.knns.append(knn)

    @staticmethod
    def _dict_to_list(iterable: Iterable):
        return iterable.values() if isinstance(iterable, dict) else iterable

    def predict(self, intermediates: Iterable, labels: torch.Tensor) -> torch.Tensor:
        """Get prediction depth for example, label pairs.

        Args:
            intermediates (Iterable): list or ordered dict of activations by layer on which to compute prediction depth.
                Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
            labels (torch.Tensor): labels to compare against KNN predictions, with shape (N,)
                In Baldock et al. (2021), these are set to the predictions of a majority of ensembled models.

        Returns:
            torch.Tensor: prediction depths of shape (N,) (note: this is NOT backprop-enabled)
        """
        knn_predictions = []
        for x, knn in zip(self._dict_to_list(intermediates), self.knns):
            predictions = knn.predict(x.reshape(x.shape[0], -1).detach().cpu().numpy())
            predictions = torch.tensor(predictions).to(dtype=labels.dtype, device=labels.device)
            knn_predictions.append(predictions)
        # dims L \times C where L is intermediate layers, C is classes
        layer_predict = torch.stack(knn_predictions, dim=0)
        match = (layer_predict == labels.broadcast_to(layer_predict.shape))
        return first_unforgettable(match)

    def __call__(self, intermediate_activations, labels):
        return self.predict(intermediate_activations, labels)


def prediction_depth(
        train_intermediates: Iterable,
        train_labels: torch.Tensor,
        test_intermediates: Iterable=None,
        test_labels: torch.Tensor=None,
        k: int=30
) -> torch.Tensor:
    """Functional equivalent of PredictionDepth. Use this function for a single call,
    whereas the PredictionDepth object is better suited for repeat calls over batches,
    since it reuses the KNN objects.

    Args:
        train_intermediates (Iterable): list or ordered dict of activations by layer for examples used to fit the KNN,
            with shape (M, ...). In Baldock et al. (2021), these are the entire training dataset.
        train_labels (torch.Tensor): labels used to fit the KNN, with shape (M,)
            In Baldock et al. (2021), these are the consensus labels of the knn_activations.
        test_intermediates (Iterable): list or ordered dict of activations by layer on which to compute prediction depth.
            Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
            If not set, use train_intermediates. Default is None.
        test_labels (torch.Tensor): labels to compare against KNN predictions, with shape (N,)
            In Baldock et al. (2021), these are set to the predictions of a majority of ensembled models.
            If not set, use train_labels. Default is None.
        k (int, optional): number of neighbours to compare in k-nearest neighbours. Defaults to 30.

    Returns:
        torch.Tensor: prediction depths of shape (N,) (note: this is NOT backprop-enabled)
    """
    pd_object = PredictionDepth(train_intermediates, train_labels, k)
    test_intermediates = train_intermediates if test_intermediates is None else test_intermediates
    test_labels = train_labels if test_labels is None else test_labels
    return pd_object(test_intermediates, test_labels)


def supervised_prototypes(representations: torch.Tensor, labels: torch.Tensor):
    """From
    Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., & Morcos, A. (2022).
    Beyond neural scaling laws: beating power law scaling via data pruning.
    Advances in Neural Information Processing Systems, 35, 19523-19536.

    Args:
        representations (torch.Tensor): per-example activations. In Sorscher et al. (2022),
            this is the embedding space of an ImageNet pre-trained self-supervised model SWaV.
            Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
        labels (torch.Tensor): true class labels of examples, with shape (N,)

    Returns:
        torch.Tensor: distance from examples to label centroids (prototypes)
            in representation space, of shape (N,)
    """
    distances = torch.empty(representations.shape[0], dtype=representations.dtype, device=representations.device)
    for i in torch.unique(labels):
        # compute centroid as mean over representations of examples
        idx = (labels == i)
        embeddings = representations[idx, ...]
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        prototype = torch.mean(embeddings, dim=0).broadcast_to(embeddings.shape)
        # for each example with this label, compute distance to correct label centroid
        distances[idx] = torch.linalg.vector_norm(embeddings - prototype, ord=2, dim=-1)
    return distances


def self_supervised_prototypes(representations: torch.Tensor, k: int=30, max_iter: int=300, random_state=None, return_kmeans_obj=False):
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
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=random_state)
    kmeans.fit(representations.detach().cpu().numpy())
    distances = kmeans.transform(representations)
    distances = torch.tensor(distances, dtype=representations.dtype, device=representations.device)
    distance_to_nearest_centroid = torch.min(distances, dim=-1).values
    if return_kmeans_obj:
        return distance_to_nearest_centroid, kmeans
    return distance_to_nearest_centroid
