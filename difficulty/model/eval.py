from collections import OrderedDict
from typing import Iterable, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def is_identity(x: torch.tensor, y: torch.tensor):
    return len(x.flatten()) == len(y.flatten()) and torch.all(x.flatten() == y.flatten())


def match_key(key: str, include: List[str] = None, exclude: List[str] = None):
    if include is not None:
        if not any(k in key for k in include):
            return False
    if exclude is not None:
        if any(k in key for k in exclude):
            return False
    return True


class SaveIntermediateHook:
    """This is used to get intermediate values in forward() pass.
    """
    def __init__(self, named_modules: Iterable[Tuple[str, nn.Module]], include: List[str]=None, exclude: List[str]=None, device='cpu', verbose=False):
        self.named_modules = list(named_modules)
        self.device = device
        self.include = include
        self.exclude = exclude
        self.verbose = verbose
        self.intermediates = OrderedDict()

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
        # copy to prevent value from changing in later operations
        if match_key(key, self.include, self.exclude):
            value = value.detach().clone().to(device=self.device)
            for k, v in self.intermediates.items():
                if is_identity(v, value):
                    if self.verbose: print(f"{key} and {k} are equal, omitting {key}")
                    return
            assert key not in self.intermediates, key
            self.intermediates[key] = value


def evaluate_intermediates(model, dataloader, device="cuda", named_modules: Iterable[Tuple[str, nn.Module]]=None, include: List[str]=None, exclude: List[str]=None, verbose=False):
    if named_modules is None:
        named_modules = model.named_modules()
    if verbose: print(model, intermediates.get_module_names(), sep="\n")
    model.eval()
    outputs = {}
    intermediates = SaveIntermediateHook(named_modules, include=include, exclude=exclude, device=device)
    with torch.no_grad():
        for i, (batch_examples, _) in enumerate(dataloader):
            with intermediates as hidden:
                if verbose: print(f"...batch {i}")
                batch_examples = batch_examples.to(device=device)
                _ = model(batch_examples)
                # Concatenate batch to output
                for k, v in hidden.items():
                    if k in outputs:
                        outputs[k] = np.concatenate([outputs[k], v.detach().cpu().numpy()])
                    else:
                        outputs[k] = v.detach().cpu().numpy()
    return outputs


def evaluate_model(model, dataloader, device="cuda"):
    modules = model.named_modules()
    (first_name, first_layer), *_ = modules  # the first module contains all others
    named_modules = [(first_name, first_layer)]
    include = first_name + ".out"  # only get the output of the containing module
    y = evaluate_intermediates(model, dataloader, device, named_modules=named_modules, include=[include])
    return y[include]


def compute_logits_for_checkpoints(ckpt_files, model, dataloader, device="cuda"):
    logits = []
    for file in tqdm(ckpt_files):
        state_dict = torch.load(file)
        model.load_state_dict(state_dict)
        logits.append(evaluate_model(model, dataloader, device=device, output_only=True))
    return np.stack(logits, axis=0)
