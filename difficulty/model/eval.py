from collections import OrderedDict
from typing import List
import numpy as np
import torch
from tqdm import tqdm


def evaluate_model(model, dataloader, device="cuda", named_modules=None, output_only=True, include=None, exclude=None, verbose=False):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i, (batch_examples, _) in enumerate(dataloader):
            batch_examples = batch_examples.to(device=device)
            y = model(batch_examples)
            y = y.detach().cpu().numpy()
            # Concatenate batch to output
            outputs.append(y)
    return np.concatenate(outputs, axis=0)


def compute_logits_for_checkpoints(ckpt_files, model, dataloader, device="cuda"):
    logits = []
    for file in tqdm(ckpt_files):
        state_dict = torch.load(file)
        model.load_state_dict(state_dict)
        logits.append(evaluate_model(model, dataloader, device=device, output_only=True))
    return np.stack(logits, axis=0)


class SaveIntermediateHook:
    """This is used to get intermediate values in forward() pass.
    """
    def __init__(self, named_modules, include=None, exclude=None, device='cpu'):
        self.module_names = OrderedDict()
        for name, module in named_modules:
            self.module_names[module] = name
            module.register_forward_hook(self)
        self.device = device
        self.include = include
        self.exclude = exclude
        self.reset()

    def reset(self):
        self.intermediates = OrderedDict()

    def get_module_names(self):
        return [x for x in self.module_names.values()]

    def __call__(self, module, args, return_val):
        layer_name = self.module_names[module]
        for arg in args:
            self._add_if_missing(layer_name + ".in", arg)
        self._add_if_missing(layer_name + ".out", return_val)

    def _add_if_missing(self, key, value):
        # copy to prevent value from changing in later operations
        value = value.detach().clone().to(device=self.device)
        for k, v in self.intermediates.items():
            if self.is_identity(v, value):
                print(f"{key} and {k} are equal, omitting {key}")
                return
        assert key not in self.intermediates
        if self._is_valid_key(key, self.include, self.exclude):
            self.intermediates[key] = value

    @staticmethod
    def is_identity(x, y):
        return len(x.flatten()) == len(y.flatten()) and torch.all(x.flatten() == y.flatten())

    @staticmethod
    def is_relu_output(x):
        return torch.all(x >= 0.).item()

    @staticmethod
    def _is_valid_key(key: str, include: List[str]=None, exclude: List[str]=None):
        if include is not None:
            if not any(k in key for k in include):
                return False
        if exclude is not None:
            if any(k in key for k in exclude):
                return False
        return True


# def evaluate_intermediates(model, dataloader, device="cuda", named_modules=None, output_only=True, include=None, exclude=None, verbose=False):
#     if output_only:
#         modules = model.named_modules()
#         *_, (last_name, last_layer) = modules
#         named_modules = [(last_name, last_layer)]
#         include = last_name + ".out"
#     else:
#         if named_modules is None:
#             named_modules = model.named_modules()
#     intermediates = SaveIntermediateHook(named_modules, include=include, exclude=exclude, device=device)
#     if verbose:
#         print(model, intermediates.get_module_names(), sep="\n")
#     model.eval()
#     outputs = {}
#     with torch.no_grad():
#         for i, (batch_examples, _) in enumerate(dataloader):
#             print(f"...batch {i}")
#             batch_examples = batch_examples.to(device=device)
#             y = model(batch_examples)
#             hidden = intermediates.get_intermediates()
#             # sanity check
#             if output_only:
#             # Concatenate batch to output
#             for k, v in hidden.items():
#                 if k in outputs:
#                     outputs[k] = np.concatenate([outputs[k], v.detach().cpu().numpy()])
#                 else:
#                     outputs[k] = v.detach().cpu().numpy()
#             # reset hook to save next batch
#             intermediates.reset()
#     return output
