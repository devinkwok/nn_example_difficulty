import unittest
import torch
import numpy.testing as npt

from difficulty.test.base import BaseTest
from difficulty.test.example_model import Model
from difficulty.metrics import *


class TestMetrics(BaseTest):

    def setUp(self) -> None:
        super().setUp()
        self.models = [Model.create() for _ in range(20)]
        out = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                out.append(m(self.data))
        self.random_logits = torch.stack(out, dim=0)
        self.consensus_labels, _ = torch.mode(torch.argmax(self.random_logits, dim=-1), dim=0)

    def test_consensus_labels(self):
        obj = OnlineConsensusLabels(device=self.device)
        for i, run in enumerate(self.random_logits):
            obj.add(run)
            if i % 5 == 0:
                obj.save(self.tmp_file)
                obj = OnlineConsensusLabels.load(self.tmp_file)
        consensus_labels = obj.get()
        self.assertEqual(consensus_labels.shape, (len(self.data),))
        self.tensors_equal(consensus_labels, self.consensus_labels)

    def test_ensemble_metrics(self):
        scores = ensemble_metrics(self.models, self.dataloader, self.device)
        acc = torch.mean((self.data_labels == torch.argmax(self.random_logits, dim=-1)).to(dtype=float), dim=0)
        self.all_close(scores["allacc"], acc)
        self.tensors_equal(scores["consensuslabel"], self.consensus_labels)
        self.tensors_equal(scores["ddd"] == 1, acc == 1)
        self.tensors_equal(scores["ddd"] == -1, acc == 0)


if __name__ == '__main__':
    unittest.main()
