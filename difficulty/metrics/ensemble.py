# metrics that are computed over multiple runs
from typing import Dict, Iterable
from pathlib import Path
import torch
import torch.nn.functional as F


from difficulty.metrics.accumulator import Accumulator, OnlineMean
from difficulty.model.eval import evaluate_model


__all__ = [
    "ensemble_metrics",
    "OnlineConsensusLabels",
    "OnlineEnsembleAccuracy",
]


def ensemble_metrics(
        models: Iterable[callable],
        dataloader: torch.utils.data.DataLoader,
        n_class: int,
        device="cuda"
    ):
    # mean accuracy, ddd
    accuracies = OnlineEnsembleAccuracy()
    consensus_labels = OnlineConsensusLabels(n_class)
    for model in models:
        eval_logits, _, acc, _ = evaluate_model(model, dataloader, device=device, return_accuracy=True)
        accuracies.add(acc)
        consensus_labels.add(eval_logits)
    #TODO conf, agr
    return {
        "ddd": accuracies.dichotomous_data_difficulty(),
        "allacc": accuracies.get(),
        "consensuslabel": consensus_labels.get(),
    }


class OnlineEnsembleAccuracy(OnlineMean):
    """Includes dichtomous data difficulty score: this is only meaningful when computed over different models.
    """

    def get_always_learned(self):
        return self.get() == 1

    def get_never_learned(self):
        return self.get() == 0

    def dichotomous_data_difficulty(self):
        """From
        Kristof Meding, Luca M. Schulze Buschoff, Robert Geirhos, & Felix A. Wichmann (2022).
        Trivial or Impossible - dichotomous data difficulty masks model differences (on ImageNet and beyond).
        In International Conference on Learning Representations.

        Returns:
            torch.Tensor: -1 if accuracies over all runs are 0, 1 if accuracies are 1, 0 otherwise.
        """
        return 1 * self.get_always_learned() - 1 * self.get_never_learned()


class OnlineConsensusLabels(Accumulator):
    def __init__(self, class_count=None, dtype=torch.long, device="cpu", metadata_lists: Dict[str, list] = {}, **metadata):
        super().__init__(dtype, device, metadata_lists, **metadata)
        self.class_count = class_count

    def save(self, file: Path):
        super().save(file, class_count=self.class_count)

    def add(self, eval_logits: torch.Tensor, dim=None, class_dim=-1, **metadata):
        super()._add(eval_logits, **metadata)
        eval_logits = eval_logits.to(device=self.device)
        predicted = F.one_hot(torch.argmax(eval_logits, dim=class_dim),
                              num_classes=eval_logits.shape[class_dim])
        predicted = torch.moveaxis(predicted, -1, class_dim)
        if dim is not None:
            assert (dim % len(eval_logits.shape)) != (class_dim % len(eval_logits.shape))
            predicted = torch.sum(predicted, dim=dim)
        if self.class_count is None:
            self.class_count = torch.zeros_like(predicted, dtype=torch.long, device=self.device)
        self.class_count += predicted
        return self

    def get(self) -> torch.Tensor:
        return torch.argmax(self.class_count, dim=-1)  # get max over one_hot dimension


#TODO mean confidence, conf in Carlini et al.

#TODO jensen shannon divergence, agr in Carlini et al.
