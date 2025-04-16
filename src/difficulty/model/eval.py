import warnings
from collections import defaultdict
from copy import deepcopy
from collections import OrderedDict
from typing import Iterable, List, Tuple, Dict, Generator, Optional
import torch
import torch.nn as nn

from difficulty.utils import ConcatTensor


def is_identity(x: torch.tensor, y: torch.tensor):
    return len(x.flatten()) == len(y.flatten()) and torch.all(x.flatten() == y.flatten())


class SaveIntermediateHook:
    def __init__(self,
        named_modules: Iterable[Tuple[str,
        nn.Module]],
        layers: List[str]=None,
        include: List[str]=None,
        exclude: List[str]=None,
        no_duplicates: bool=True,
        device='cpu',
        verbose=False
    ):
        """Get intermediate values (output of every module/layer)
        from a forward() pass, by hooking into nn.Module.

        This is a context manager which resets and removes all hooks on exit.
        Layers with identical intermediate values are ignored.

        Example usage:
            ```
            intermediates = SaveIntermediateHook(model.named_modules())
            for x in data:
                with intermediates as hidden:
                    model(x)
                    yield hidden
            ```

        Args:
            named_modules (Iterable[Tuple[str, nn.Module]]): modules to add hook to.
            layers (Set[str], optional): If set, only include these exact layer names. A layer name
                is the module name followed by ".in" or ".out" indicating the input or output to the module.
                If set, ignores include, exclude, and assumes no_duplicates=False. Defaults to None.
            include (List[str], optional): If set, only include modules with names
                containing at least one of these patterns. Defaults to None.
            exclude (List[str], optional): If set, exclude any modules with names
                containing any of these patterns. Defaults to None.
            no_duplicates (bool): If True, do not save any copies of the same layer
                (comparison ignores shape, but preserves dim order). Defaults to True.
            device (str, optional): Device to move intermediates to. Defaults to 'cpu'.
            verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
                This occurs often when modules are nested. Defaults to False.
        """
        self.device = device
        if layers is not None:
            self.layers = set(layers)
            self.named_modules = self.filter_modules(layers, list(named_modules))
            self.include = None
            self.exclude = None
            self.no_duplicates = False
        else:
            self.layers = None
            self.named_modules = list(named_modules)
            self.include = include
            self.exclude = exclude
            self.no_duplicates = no_duplicates
        self.verbose = verbose
        self.intermediates = OrderedDict()

    @staticmethod
    def filter_modules(layers, named_modules):
        # remove .in and .out from layer names, return modules with these names
        module_names = set(x[:-3] if x.endswith(".in") else x[:-4] for x in layers)
        return [(k, v) for k, v in named_modules if k in module_names]

    def __enter__(self):
        self.module_names = OrderedDict()
        self.handles = []
        for name, module in self.named_modules:
            self.module_names[module] = name
            self.handles.append(module.register_forward_hook(self))
        return self.intermediates

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for handle in self.handles:
            handle.remove()
        self.intermediates = OrderedDict()

    def __call__(self, module, args, return_val):
        layer_name = self.module_names[module]
        for arg in args:
            self._add_if_missing(layer_name + ".in", arg)
        self._add_if_missing(layer_name + ".out", return_val)

    def _add_if_missing(self, key, value):
        # copy value to prevent it from changing in later operations
        if self.layers is None:
            if match_key(key, self.include, self.exclude):
                value = value.detach().clone().to(device=self.device)
                if self.no_duplicates:
                    for k, v in self.intermediates.items():
                        if is_identity(v, value):
                            if self.verbose: warnings.warn(f"{key} and {k} are equal, omitting {key}")
                            return
                assert key not in self.intermediates, key
                self.intermediates[key] = value
        elif key in self.layers:  # ignore include, exclude, and no_duplicates
            value = value.detach().clone().to(device=self.device)
            assert key not in self.intermediates, key
            self.intermediates[key] = value


def match_key(key: str, include: List[str] = None, exclude: List[str] = None):
    if include is not None:
        if not any(k in key for k in include):
            return False
    if exclude is not None:
        if any(k in key for k in exclude):
            return False
    return True


def find_intermediate_layers(
        model: nn.Module,
        input_shape: Tuple[int],
        n_test_points: int=100,
        dtype=torch.float32,
        device: str="cuda",
        named_modules: Iterable[Tuple[str, nn.Module]]=None,
        include: List[str]=None,
        exclude: List[str]=None,
        verbose=False,
    ):
    """Use random data to determine which layers have distinct intermediate values in a model.

    Args:
        model (nn.Module): Model to evaluate.
        input_shape (Tuple[int]): shape of input tensor, excluding batch dimension.
        n_test_points (int): how many random data points to generate as input.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        named_modules (Iterable[Tuple[str, nn.Module]], optional): If set,
            only get intermediates values from these modules,
            otherwise include all intermediates from model.named_modules(). Defaults to None.
        include (List[str], optional): If set, only include modules with names
            containing at least one of these patterns. Defaults to None.
        exclude (List[str], optional): If set, exclude any modules with names
            containing any of these patterns. Defaults to None.
        device (str, optional): Device to move intermediates to. Defaults to 'cpu'.
        verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
            This occurs often when modules are nested. Defaults to False.
    """
    with torch.no_grad():
        if named_modules is None:  # model.named_modules() returns generator, make into list so it's reusable
            named_modules = list(model.named_modules())
        random_input = torch.randn((n_test_points, *input_shape), dtype=dtype, device=device)

        model.to(device=device)
        model.eval()
        intermediates = SaveIntermediateHook(
            named_modules, include=include, exclude=exclude, no_duplicates=True, device=device, verbose=verbose)
        with intermediates as hidden:
            model(random_input)
        layers = list(hidden.keys())

        # check that the list of layer names and modules gives same result
        intermediates = SaveIntermediateHook(
            named_modules, layers=layers, device=device, verbose=verbose)
        with intermediates as hidden_2:
            model(random_input)
        assert set(hidden.keys()) == set(hidden_2.keys()), [set(hidden.keys()), set(hidden_2.keys())]
        for k, v in hidden.items():
            assert torch.allclose(v, hidden_2[k])

        return layers


def batch_evaluate_intermediates(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layers: List[str],
        device: str="cuda",
        verbose=False,
) -> Generator:
    """Evaluate a model on a dataloader, returning inputs, intermediate values, outputs, and labels.

    To get layers and named_modules, use `find_intermediate_layers()`.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
        layers (List[str], optional): Only include these exact layer names. A layer name
            is the module name followed by ".in" or ".out" indicating the input or output to the module.
            If set, ignores include, exclude, and assumes no_duplicates=False. Defaults to None.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
            This occurs often when modules are nested. Defaults to False.

    Yields:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            a tuple for every batch containing (inputs, intermediate values, outputs, true labels)
    """
    model.to(device=device)
    model.eval()
    intermediates = SaveIntermediateHook(
        model.named_modules(), layers=layers, device=device, verbose=verbose)
    with torch.no_grad():
        for batch_examples, labels in dataloader:
            with intermediates as hidden:
                batch_examples = batch_examples.to(device=device)
                labels = labels.to(device=device)
                output = model(batch_examples)
                yield batch_examples, hidden, output, labels


def combine_batches(eval_intermediates_generator: Generator, n_examples=None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Combines batches from batch_evaluate_intermediates().

    Args:
        eval_intermediates_generator: the generator returned by batch_evaluate_intermediates().
        n_examples (int, optional): if set, preallocate memory for concatenating tensors.
            Total number of examples in dim 0 over all batches. Default is None.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            tuple containing (inputs, intermediate values, outputs, true labels)
    """
    inputs, outputs, labels = ConcatTensor(n_examples), ConcatTensor(n_examples), ConcatTensor(n_examples)
    hiddens = defaultdict(lambda: ConcatTensor(n_examples))
    for input, hidden, output, label in eval_intermediates_generator:
        inputs.append(input)
        outputs.append(output)
        labels.append(label)
        for k, v in hidden.items():
            hiddens[k].append(v)

    inputs = inputs.cat()
    outputs = outputs.cat()
    labels = labels.cat()
    for k, v in hiddens.items():
        hiddens[k] = hiddens[k].cat()
    return inputs, hiddens, outputs, labels


def split_batches(inputs, intermediates, outputs, labels, batch_size):
    inputs = torch.split(inputs, batch_size, dim=0)
    intermediates = {k: torch.split(v, batch_size, dim=0) for k, v in intermediates.items()}
    outputs = torch.split(outputs, batch_size, dim=0)
    labels = torch.split(labels, batch_size, dim=0)
    for i, (input, output, label) in enumerate(zip(inputs, outputs, labels)):
        intermediate = {k: v[i] for k, v in intermediates.items()}
        yield input, intermediate, output, label


def evaluate_intermediates(model: nn.Module, dataloader: torch.utils.data.DataLoader, layers: List[str], device: str="cuda", verbose=False) -> torch.Tensor:
    """Get all intermediates for data in dataloader (combine all batches together).

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
        layers (List[str], optional): Only include these exact layer names. A layer name
            is the module name followed by ".in" or ".out" indicating the input or output to the module.
            If not set, use all layers found by `find_intermediate_layers`. Defaults to None.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        verbose (bool, optional): Warn when two intermediates are identical and one is discarded.
            This occurs often when modules are nested. Defaults to False.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            same output as batch_evaluate_intermediates, but with batches combined to shape (N, ...)
    """
    n_examples = len(dataloader.dataset)
    generator = batch_evaluate_intermediates(model, dataloader, layers, device=device, verbose=verbose)
    inputs, intermediates, outputs, labels = combine_batches(generator, n_examples=n_examples)
    return inputs, intermediates, outputs, labels


def evaluate_model(model: nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   state_dict: Dict=None,
                   device: str="cuda",
                   return_accuracy: bool=False,
                   loss_fn: nn.Module=None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Evaluate model and return outputs, and optionally labels, accuracy, and loss.
    This does not return intermediate values.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to evaluate on.
        state_dict (Dict, optional): If set, load these model parameters before evaluating. Defaults to None.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        return_accuracy (bool, optional): If True, include
            `torch.argmax(outputs) == labels` in return. Defaults to False.
        loss_fn (nn.Module, optional): If set, include
            `loss_fn(outputs, labels)` in return. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            a tuple of (outputs, true labels, accuracy, loss), with first dimension over examples in batch order.
            If return_accuracy or loss_fn are not set, accuracy or loss are None respectively.
    """

    if state_dict is not None:
        model = deepcopy(model)
        model.load_state_dict(state_dict)
    # set layers=[] because no intermediates needed
    _, _, outputs, labels = evaluate_intermediates(model, dataloader, [], device=device)
    acc = torch.argmax(outputs, dim=-1) == labels if return_accuracy else None
    loss = None if loss_fn is None else loss_fn(outputs, labels)
    return outputs, labels, acc, loss


def batch_evaluate_model(model: nn.Module,
                   batch: torch.Tensor,
                   labels: torch.Tensor,
                   state_dict: Dict=None,
                   device: str="cuda",
                   return_accuracy: bool=False,
                   loss_fn: nn.Module=None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Evaluate model and return outputs, and optionally labels, accuracy, and loss.
    This does not return intermediate values.

    Args:
        model (nn.Module): Model to evaluate.
        batch (torch.Tensor): Data to input to  model.
        labels (torch.Tensor): target for loss function.
        state_dict (Dict, optional): If set, load these model parameters before evaluating. Defaults to None.
        device (str, optional): Device to evaluate on. Defaults to "cuda".
        return_accuracy (bool, optional): If True, include
            `torch.argmax(outputs) == labels` in return. Defaults to False.
        loss_fn (nn.Module, optional): If set, include
            `loss_fn(outputs, labels)` in return. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            a tuple of (outputs, true labels, accuracy, loss), with first dimension over examples in batch order.
            If return_accuracy or loss_fn are not set, accuracy or loss are None respectively.
    """
    if state_dict is not None:
        model = deepcopy(model)
        model.load_state_dict(state_dict)

    model.to(device=device)
    model.eval()
    batch = batch.to(device=device)
    labels = labels.to(device=device)
    outputs = model(batch)
    acc = torch.argmax(outputs, dim=-1) == labels if return_accuracy else None
    loss = None if loss_fn is None else loss_fn(outputs, labels)
    return outputs, labels, acc, loss
