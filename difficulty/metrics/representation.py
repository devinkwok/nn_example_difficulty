from itertools import chain
from typing import List, Iterable, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from difficulty.metrics.pointwise import pointwise_metrics
from difficulty.metrics.forget import first_unforgettable

from difficulty.utils import detach_tensors, Stopwatch, ConcatTensor
from difficulty.model.eval import evaluate_intermediates, find_intermediate_layers, combine_batches


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
        dtype=torch.float64,
        generate_pointwise_metrics=False,
        verbose=False,
        n_examples: int=None,
        pd_layers: List[str]=None,
        pd_k: int=30,
        pd_append_softmax: bool=False,
        pd_train_labels: torch.Tensor=None,
        pd_test_dataloader: torch.utils.data.DataLoader=None,
        pd_return_layerpred=False,
        proto_layer: str=None,
        selfproto_k: int=30,
        selfproto_random_state: int=None,
        selfproto_max_iter: int=300,
    ):
        """Generate representation metrics: prediction_depth, supervised_prototypes, and self_supervised_prototypes.

        Args:
            model (nn.Module): Model to evaluate.
            dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
            device (str, optional): Device to evaluate on. Defaults to "cuda".
            to_cpu (bool, optional): if results should be moved to cpu. Defaults to True.
            to_numpy (bool, optional): if results should be converted to numpy arrays. Defaults to False.
            generate_pointwise_metrics (bool, optional):
                whether to also call pointwise_metrics (to avoid re-evaluating model). Defaults to False.
            verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
                This occurs often when modules are nested. Defaults to False.
            n_examples (int, optional): if set, preallocate memory for concatenating tensors.
                Total number of examples in dim 0 over all batches. Default is None.
            pd_layers (List[str], optional): Only include these exact layer names. A layer name
                is the module name followed by ".in" or ".out" indicating the input or output to the module.
                If not set, use all layers found by `find_intermediate_layers`. Defaults to None.
            pd_k (int, optional): for prediction depth,
                number of neighbours to compare in k-nearest neighbours. Defaults to 30.
            pd_append_softmax (bool, optional): for prediction depth,
                whether to include softmax of outputs as layer. Defaults to False.
            pd_train_labels (torch.Tensor, optional): for prediction depth, labels used to fit the KNN, with shape (M,)
                In Baldock et al. (2021), these are the consensus labels of the knn_activations.
                If None, use the labels in the dataloader. Defaults to None.
            pd_test_dataloader (torch.utils.data.DataLoader, optional):
                Compute prediction depth for this data instead of the dataloader (the dataloader is used to fit the KNN only).
                If None, use the dataloader to fit the KNN and compute prediction depth. Defaults to None.
            pd_return_layerpred (bool, optional): If True, return each layer's predicted class as a score
                with the key "pd_{k}" for the kth layer. Defaults to False.
            proto_layer (str, optional): for supervised and self-supervised prototypes, name of layer to use as representations.
                If None, use last layer of network found by `find_intermediate_layers` (excluding softmax). Defaults to None.
            selfproto_k (int, optional): for self-supervised prototypes,
                number of means for k-means clustering. Sorscher et al. (2022)
                recommends any value within an order of magnitude of the true number classes,
                noting that the performance when using this metric for data pruning
                is not very sensitive to k. Defaults to 30.
            selfproto_random_state (int, optional): for self-supervised prototypes,
                deterministic seed for initializing k-means clustering. Defaults to None.
            selfproto_max_iter (int, optional): for self-supervised prototypes,
                number of k-means clustering iterations to run. Defaults to 300.

        Returns:
            Dict[str, torch.Tensor]: dictionary of representation metrics, and optionally pointwise metrics.
        """
        if verbose:
            stopwatch = Stopwatch("representation_metrics")
            stopwatch.start()
        batch_shape = next(iter(dataloader))[0].shape
        if pd_layers is None:  # include all layers
            pd_layers = find_intermediate_layers(model, batch_shape[1:], device=device)
        # include proto_layer to get intermediates for both prediction depth and (self) supervised prototypes
        append_proto_layer = (pd_layers is not None) and (proto_layer is not None) and (proto_layer not in pd_layers)

        proto_layers = [proto_layer] if append_proto_layer else []
        generator = evaluate_intermediates(model, dataloader, pd_layers + proto_layers, device=device, verbose=verbose)
        _, intermediates, outputs, labels = combine_batches(generator, n_examples=n_examples)
        if verbose: stopwatch.lap("evaluate_intermediates")
        metrics = {}

        # generate other metrics here using model eval outputs
        if generate_pointwise_metrics:
            metrics = pointwise_metrics(outputs, labels, to_cpu=to_cpu, to_numpy=to_numpy)
        if verbose: stopwatch.lap("pointwise_metrics")

        # remove proto_layer from intermediates if it shouldn't be included
        pd_intermediates = dict(intermediates)
        if append_proto_layer:
            del pd_intermediates[proto_layer]
        # use true labels as consensus labels if not set
        train_labels = labels if pd_train_labels is None else pd_train_labels
        pd_obj = PredictionDepth(pd_intermediates, train_labels,
                                 outputs if pd_append_softmax else None,
                                 k=pd_k, verbose=verbose)
        if verbose: stopwatch.lap("PredictionDepth fit KNNs")

        # compute prediction depth on knn training data if test data not provided
        if pd_test_dataloader is None:  # memory isn't an issue and batching is slower, so just do all examples at once
            test_generator = [(None, pd_intermediates, outputs, train_labels)]
        else:  # use pd_layers so that we don't include proto_layer
            test_generator = evaluate_intermediates(model, pd_test_dataloader, pd_layers, device=device, verbose=verbose)

        pd = ConcatTensor(n_examples)
        layerpred = [ConcatTensor(n_examples) for _ in range(pd_obj.n_layers)] if pd_return_layerpred else None
        for _, test_intermediates, test_outputs, test_labels in test_generator:
            test_outputs = test_outputs if pd_append_softmax else None
            knn_layers = pd_obj.knn_predict(test_intermediates, output_for_softmax=test_outputs)
            if pd_return_layerpred:
                for i, layer in enumerate(knn_layers):
                    layerpred[i].append(layer)
            pd.append(pd_obj.depth_from_knn_predict(knn_layers, test_labels))
        pd = pd.cat()
        if pd_return_layerpred:
            layerpred = {f"pdlayer{i}": x.cat() for i, x in enumerate(layerpred)}
            metrics = {**metrics, **detach_tensors(layerpred, to_cpu=to_cpu, to_numpy=to_numpy)}
        if verbose: stopwatch.lap("PredictionDepth scores")

        # for prototypes, use last layer as representation by default
        if proto_layer is None:
            proto_layer = pd_layers[-1]
        representations = intermediates[proto_layer]
        # make higher precision
        representations = representations.to(dtype=dtype)
        proto = supervised_prototypes(representations, labels)
        if verbose: stopwatch.lap("supervised_prototypes")
        selfproto = self_supervised_prototypes(representations, k=selfproto_k, max_iter=selfproto_max_iter, random_state=selfproto_random_state)
        if verbose:
            stopwatch.stop("self_supervised_prototypes")

        metrics = {**metrics, **detach_tensors(
            {
                "pd": pd,
                "proto": proto,
                "selfproto": selfproto,
            }, to_cpu=to_cpu, to_numpy=to_numpy)
        }
        return metrics


class PredictionDepth:
    def __init__(self,
            intermediate_activations: Iterable,
            consensus_labels: torch.Tensor,
            output_for_softmax: Optional[torch.Tensor]=None,
            k: int=30,
            verbose=False,
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
            output_for_softmax (Optional[torch.Tensor]): if set, output tensor to compute
                softmax over and append to intermediate activations. Defaults to None.
            k (int, optional): number of neighbours to compare in k-nearest neighbours. Defaults to 30.
            verbose (bool, optional): print layer number that is being completed. Defaults to False.
        """
        self.include_softmax = (output_for_softmax is not None)
        labels = consensus_labels.detach().cpu().numpy()
        self.knns = []
        self.verbose = verbose
        intermediates = self._list_and_softmax(intermediate_activations, output_for_softmax)
        for i, x in enumerate(intermediates):
            if self.verbose:
                print(f"pd: fitting KNN to layer {i}")
            knn = KNeighborsClassifier(k)  # use standard Euclidean distance
            x = x.reshape(x.shape[0], -1).detach().cpu().numpy()
            knn.fit(x, labels)
            self.knns.append(knn)

    @property
    def n_layers(self):
        return len(self.knns)

    @staticmethod
    def _list_and_softmax(intermediates: torch.Tensor, output_for_softmax: Optional[torch.Tensor]):
        intermediates = intermediates.values() if isinstance(intermediates, dict) else intermediates
        if output_for_softmax is not None:
            return chain(intermediates, [torch.nn.functional.softmax(output_for_softmax, dim=-1)])
        return intermediates

    def knn_predict(self, intermediates: Iterable, output_for_softmax: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        """Get prediction at each layer for example, label pairs.

        Args:
            intermediates (Iterable): list or ordered dict of L activations by layer on which to compute prediction depth.
                Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
            output_for_softmax (torch.Tensor, optional): If set, include softmax of outputs as last layer.
                Specifically, compute softmax over the last dimension of this tensor,
                and append to intermediates. Defaults to None.

        Returns:
            torch.Tensor: boolean prediction accuracies of shape (L, N)
        """
        assert (output_for_softmax is None) == (False if self.include_softmax else True)
        knn_predictions = []
        intermediates = self._list_and_softmax(intermediates, output_for_softmax)
        for i, (x, knn) in enumerate(zip(intermediates, self.knns)):
            if self.verbose:
                print(f"pd: evaluating KNN for layer {i}")
            predictions = knn.predict(x.reshape(x.shape[0], -1).detach().cpu().numpy())
            predictions = torch.tensor(predictions)
            knn_predictions.append(predictions)
        return knn_predictions

    @staticmethod
    def depth_from_knn_predict(layer_predict: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Helper function that takes output of knn_predict and returns prediction depth.

        Args:
            layer_predict (List[torch.Tensor]): Output of knn_predict: a list of class predictions per layer of shape (N,).
            labels (torch.Tensor): labels to compare against KNN predictions, with shape (N,)
                In Baldock et al. (2021), these are set to the predictions of a majority of ensembled models.

        Returns:
            torch.Tensor: prediction depths of shape (N,) (note: this is NOT backprop-enabled)
        """
        # dims L \times C where L is intermediate layers, C is classes
        layer_predict = torch.stack(layer_predict, dim=0).to(dtype=labels.dtype, device=labels.device)
        match = (layer_predict == labels.broadcast_to(layer_predict.shape))
        return first_unforgettable(match)

    def predict(self, intermediates: Iterable, labels: torch.Tensor, output_for_softmax: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Get prediction depth for example, label pairs. Equivalent to calling
        `PredictionDepth.depth_from_knn_predict(pd_object.knn_predict(intermediates, output_for_softmax), labels)`.

        Args:
            intermediates (Iterable): list or ordered dict of activations by layer on which to compute prediction depth.
                Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
            labels (torch.Tensor): labels to compare against KNN predictions, with shape (N,)
                In Baldock et al. (2021), these are set to the predictions of a majority of ensembled models.
            output_for_softmax (torch.Tensor, optional): If set, include softmax of outputs as last layer.
                Specifically, compute softmax over the last dimension of this tensor,
                and append to intermediates. Defaults to None.

        Returns:
            torch.Tensor: prediction depths of shape (N,) (note: this is NOT backprop-enabled)
        """
        layer_predict = self.knn_predict(intermediates, output_for_softmax=output_for_softmax)
        return self.depth_from_knn_predict(layer_predict, labels)

    def __call__(self, intermediate_activations, labels, output_for_softmax=None):
        return self.predict(intermediate_activations, labels, output_for_softmax)


def prediction_depth(
        train_intermediates: Iterable,
        train_labels: torch.Tensor,
        test_intermediates: Iterable=None,
        test_labels: torch.Tensor=None,
        train_outputs: Optional[torch.Tensor]=None,
        test_outputs: Optional[torch.Tensor]=None,
        k: int=30,
        verbose=False,
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
        train_outputs (Optional[torch.Tensor]): if set, output tensor to compute
            softmax over and append to train_intermediates. Defaults to None.
        test_outputs (Optional[torch.Tensor]): if set, output tensor to compute
            softmax over and append to test_intermediates. Defaults to None.
        k (int, optional): number of neighbours to compare in k-nearest neighbours. Defaults to 30.
        verbose (bool, optional): print layer number that is being completed. Defaults to False.

    Returns:
        torch.Tensor: prediction depths of shape (N,) (note: this is NOT backprop-enabled)
    """
    pd_object = PredictionDepth(train_intermediates, train_labels, output_for_softmax=train_outputs, k=k, verbose=verbose)
    test_intermediates = train_intermediates if test_intermediates is None else test_intermediates
    test_labels = train_labels if test_labels is None else test_labels
    test_outputs = train_outputs if test_outputs is None else test_outputs
    return pd_object(test_intermediates, test_labels, output_for_softmax=test_outputs)


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
        k (int, optional): number of means for k-means clustering.
            Sorscher et al. (2022) recommends any value within an order of magnitude of the true number classes,
            noting that the performance when using this metric for data pruning is not very sensitive to k. Defaults to 30.
        max_iter (int, optional): number of k-means clustering iterations to run. Defaults to 300.

    Returns:
        torch.Tensor: distance from examples to cluster centroids (self-supervised prototypes)
            in representation space, of shape (N,)
    """
    dtype = representations.dtype
    device = representations.device
    representations = representations.reshape(representations.shape[0], -1).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=random_state)
    kmeans.fit(representations)
    distances = kmeans.transform(representations)
    distances = torch.tensor(distances, dtype=dtype, device=device)
    distance_to_nearest_centroid = torch.min(distances, dim=-1).values
    if return_kmeans_obj:
        return distance_to_nearest_centroid, kmeans
    return distance_to_nearest_centroid
