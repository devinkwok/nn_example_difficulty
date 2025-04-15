from typing import List, Iterable, Generator, Union, Dict
import warnings
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from difficulty.metrics.pointwise import pointwise_metrics
from difficulty.metrics.forget import first_unforgettable

from difficulty.utils import detach_tensors, Stopwatch
from difficulty.model.eval import find_intermediate_layers, evaluate_intermediates


__all__ = [
    "representation_metrics",
    "intermediates_iterable",
    "knn_predict",
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
        pd_layers: List[str]=None,
        pd_k: int=30,
        pd_append_softmax: bool=False,
        pd_train_labels: torch.Tensor=None,
        pd_test_dataloader: torch.utils.data.DataLoader=None,
        pd_return_layerpred=False,
        use_faiss: bool=False,
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
            use_faiss (bool, optional): If True, use Faiss library to do K-nearest-neighbor search (faster). Defaults to False.
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

        if pd_layers is None or proto_layer is None:  # automatically search for all layers
            batch_shape = next(iter(dataloader))[0].shape
            all_layers = find_intermediate_layers(model, batch_shape[1:], device=device)

        #### prototypes
        # use last layer as representation by default
        proto_layer = all_layers[-1] if proto_layer is None else proto_layer
        _, representations, outputs, labels = evaluate_intermediates(model, dataloader, [proto_layer], device=device, verbose=verbose)
        # make higher precision
        representations = representations[proto_layer].to(dtype=dtype)

        proto = supervised_prototypes(representations, labels)
        if verbose: stopwatch.lap("supervised_prototypes")

        selfproto = self_supervised_prototypes(representations, k=selfproto_k, max_iter=selfproto_max_iter, random_state=selfproto_random_state)
        if verbose: stopwatch.stop("self_supervised_prototypes")

        metrics = {**detach_tensors({
                "proto": proto,
                "selfproto": selfproto,
            }, to_cpu=to_cpu, to_numpy=to_numpy)
        }

        #### prediction depth
        # include all layers if no particular layers specified
        pd_layers = all_layers if pd_layers is None else pd_layers

        train_intermediates = intermediates_iterable(
            model, dataloader, pd_layers, device=device, verbose=verbose, append_softmax=pd_append_softmax)
        # use true labels as consensus labels if not set
        train_labels = labels if pd_train_labels is None else pd_train_labels

        # compute prediction depth on knn training data if test data not provided
        test_intermediates = None if pd_test_dataloader is None else intermediates_iterable(
            model, pd_test_dataloader, pd_layers, device=device, verbose=verbose, append_softmax=pd_append_softmax)
        test_labels = None if pd_test_dataloader is None else torch.cat([labels for _, labels in pd_test_dataloader], dim=0)

        pd, knn_outputs = prediction_depth(
            train_intermediates, train_labels, test_intermediates, test_labels, k=pd_k, verbose=verbose, return_matches=True, use_faiss=use_faiss, device=device)
        if verbose: stopwatch.lap("prediction_depth")

        metrics = {**metrics, **detach_tensors({"pd": pd}, to_cpu=to_cpu, to_numpy=to_numpy)}
        if pd_return_layerpred:
            knn_outputs = {f"pdlayer{i}": x for i, x in enumerate(knn_outputs)}
            metrics = {**metrics, **detach_tensors(knn_outputs, to_cpu=to_cpu, to_numpy=to_numpy)}

        #### other metrics
        if generate_pointwise_metrics:
            metrics = {**metrics, **pointwise_metrics(outputs, labels, to_cpu=to_cpu, to_numpy=to_numpy)}
        if verbose: stopwatch.lap("pointwise_metrics")

        return metrics


def intermediates_iterable(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layers: List[str],
        device: str="cuda",
        verbose=False,
        append_softmax=False,
        layers_per_run=1,
) -> Generator[torch.Tensor, None, None]:
    """Generates intermediate values for an entire dataloader, one layer at a time.
    Inference is run once every layers_per_run layers,
    this is because prediction depth fits a KNN per layer
    to all intermediates from the training data which is memory intensive.
    By iterating over only a few layers at a time, previous results can be discarded after
    fitting/predicting KNNs, reducing memory use.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
        layers (List[str], optional): Only include these exact layer names. A layer name
            is the module name followed by ".in" or ".out" indicating the input or output to the module.
            If not set, use all layers found by `find_intermediate_layers`. Defaults to None.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
            This occurs often when modules are nested. Defaults to False.
        append_softmax (bool, optional): whether to include softmax of outputs as layer. Defaults to False.
        layers_per_run (int, optional): number of layers per inference run. Defaults to 1.

    Yields:
        torch.Tensor: intermediate values for each layer, of shape (N, ...)
    """
    for i in range(0, len(layers), layers_per_run):
        _, intermediates, outputs, _ = evaluate_intermediates(model, dataloader, layers[i:min(i+layers_per_run, len(layers))], device=device, verbose=verbose)
        for value in intermediates.values():
            yield value
    if append_softmax:
        yield torch.nn.functional.softmax(outputs, dim=-1)


def knn_predict(
        train_x: torch.Tensor,
        train_labels: torch.Tensor,
        test_x: torch.Tensor=None,
        test_labels: torch.Tensor=None,
        k: int=30,
        use_faiss=False,
        device="cuda",
        verbose=False,
) -> torch.Tensor:
    """Use K nearest neighbours to predict labels from test_x using train_x and train_labels

    Args:
        train_x (torch.Tensor): data used to fit KNN, of shape (N, ...).
        train_labels (torch.Tensor): target labels for train_x, of shape (N,).
        test_x (torch.Tensor, optional): data on which to predict classes, of shape (M, ...).
            If None, return predictions on train_x. Defaults to None.
        test_labels (torch.Tensor, optional): labels for test_x to compare with predictions, of shape (M,).
            If None, use train_labels. Must be set if test_x is not None. Defaults to None.
        k (int, optional): number of neighbours to use in KNN. Defaults to 30.
        use_faiss (bool, optional): If True, use Faiss library to do K-nearest-neighbor search (faster). Defaults to False.
        device (str, optional): If use_faiss, use this device to evaluate on. Defaults to "cuda".
        verbose (bool, optional): print if using Faiss. Defaults to False.

    Returns:
        torch.Tensor: for each test_x, whether prediction matches test_labels is correct, of shape (M,)
    """
    # use standard Euclidean distance
    if use_faiss and verbose:
        warnings.warn(f"Using Faiss to compute K-nearest neighbors, device={device}")
        from difficulty.faiss_knn import FaissKNeighbors
        knn = FaissKNeighbors(k, device=device)
    else:
        knn = KNeighborsClassifier(k)
    train_x = train_x.reshape(train_x.shape[0], -1).detach().cpu().numpy()
    knn.fit(train_x, train_labels.detach().cpu().numpy())
    if test_x is None:
        assert test_labels is None
        test_x = train_x
        test_labels = train_labels
    else:
        assert test_labels is not None
        test_x = test_x.reshape(test_x.shape[0], -1).detach().cpu().numpy()
    predictions = torch.tensor(knn.predict(test_x))
    match = predictions == test_labels.detach().cpu()
    return match


def prediction_depth(
        train_intermediates: Union[Iterable, Dict],
        train_labels: torch.Tensor,
        test_intermediates: Union[Iterable, Dict]=None,
        test_labels: torch.Tensor=None,
        k: int=30,
        verbose=False,
        return_matches=False,
        use_faiss=False,
        device="cuda",
) -> torch.Tensor:
    """Functional equivalent of PredictionDepth. Use this function for a single call,
    whereas the PredictionDepth object is better suited for repeat calls over batches,
    since it reuses the KNN objects.

    Args:
        train_intermediates (Union[Iterable, Dict]): iterable or ordered dict of L activations
            in order from input towards output layers of examples used to fit the KNN,
            with shape (M, ...). In Baldock et al. (2021), these are the entire training dataset.
        train_labels (torch.Tensor): labels used to fit the KNN, with shape (M,)
            In Baldock et al. (2021), these are the consensus labels of the knn_activations.
        test_intermediates (Union[Iterable, Dict]): iterable or ordered dict of L activations
            in order from input towards output layers of examples on which to compute prediction depth.
            Activations have shape (N, ...) where N is number of examples, and remaining dimensions are flattened.
            If not set, use train_intermediates. Default is None.
        test_labels (torch.Tensor): labels to compare against KNN predictions, with shape (N,)
            In Baldock et al. (2021), these are set to the predictions of a majority of ensembled models.
            If not set, use train_labels. Default is None.
        k (int, optional): number of neighbours to compare in k-nearest neighbours. Defaults to 30.
        verbose (bool, optional): print layer number that is being completed. Defaults to False.
        return_matches (bool, optional): for each layer and example in the N outputs,
            return whether it matches the label, with shape (L, N). Defaults to False.
        use_faiss (bool, optional): If True, use Faiss library to do K-nearest-neighbor search (faster). Defaults to False.
        device (str, optional): Device to evaluate on. Defaults to "cuda".

    Returns:
        torch.Tensor: prediction depths of shape (N,) (note: this is NOT backprop-enabled),
            if return_matches=True, return (prediction_depth, matches),
            where matches is a torch.Tensor indicating whether the KNN prediction matches the label
            for each layer and example in the N outputs, with shape (L, N).
    """
    if isinstance(train_intermediates, dict):
        train_intermediates = train_intermediates.values()
    if isinstance(test_intermediates, dict):
        test_intermediates = test_intermediates.values()

    # compute knns per layer
    knn_predictions = []

    # if test not set, run predict on train
    if test_intermediates is None:
        assert test_labels is None
        for i, train_x in enumerate(train_intermediates):
            if verbose:
                print(f"pd: fitting KNN to layer {i}")
            knn_predictions.append(knn_predict(train_x, train_labels, k=k, use_faiss=use_faiss, device=device, verbose=verbose))
    else:
        for i, (train_x, test_x) in enumerate(zip(train_intermediates, test_intermediates)):
            if verbose:
                print(f"pd: fitting KNN to layer {i}")
            knn_predictions.append(knn_predict(
                train_x, train_labels, test_x, test_labels, k=k, use_faiss=use_faiss, device=device, verbose=verbose))

    matches = torch.stack(knn_predictions, dim=0)
    matches = matches.to(dtype=train_labels.dtype, device=train_labels.device)
    pd = first_unforgettable(matches)
    if return_matches:
        return pd, matches
    return pd


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
        selfproto_random_state (int, optional): deterministic seed for initializing k-means clustering. Defaults to None.

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
