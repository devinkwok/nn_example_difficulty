import unittest
import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn


from open_lth.models import cifar_resnet
from open_lth.api import get_hparams_dict
from difficulty.model.eval import evaluate_model, evaluate_intermediates
from difficulty.metrics.representation import *


class TestModel(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.batch_size = 4
        self.data = torch.randn([self.n, 3, 9, 9])
        self.labels = np.concatenate([np.zeros(self.n // 2), np.ones(self.n - self.n // 2)])
        dataset = torch.utils.data.TensorDataset(self.data, torch.zeros(self.data.shape[0]))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, drop_last=False)
        self.model = nn.Sequential(
                        nn.Conv2d(3, 10, 3),
                        cifar_resnet.Model.Block(10, 5, downsample=True)
                    )

    def test_get_hparams(self):
        hparams = get_hparams_dict("./difficulty/test")
        self.assertTrue("Dataset" in hparams)
        self.assertTrue("Model" in hparams)
        self.assertTrue("Pretraining Dataset" in hparams)
        self.assertTrue("Pretraining Training" in hparams)
        self.assertTrue("Pruning" in hparams)
        self.assertTrue("Training" in hparams)

    def test_evaluate_intermediates(self):
        y = evaluate_intermediates(self.model, self.dataloader, device="cpu")
        npt.assert_array_equal(list(y.keys()), 
            ['0.in', '0.out', '1.conv1.out', '1.bn1.out', '1.relu1.out', '1.conv2.out', '1.bn2.out', '1.shortcut.0.out', '1.shortcut.1.out', '1.relu2.in', '1.relu2.out'])
        [self.assertEqual(len(v), self.n) for v in y.values()]
        *_, last_module = self.model.named_modules()
        y = evaluate_intermediates(self.model, self.dataloader, device="cpu", named_modules=[last_module])
        npt.assert_array_equal(list(y.keys()), ['1.shortcut.1.in', '1.shortcut.1.out'])
        [self.assertEqual(len(v), self.n) for v in y.values()]
        y = evaluate_intermediates(self.model, self.dataloader, device="cpu", include=["conv"])
        npt.assert_array_equal(list(y.keys()), ['1.conv1.in', '1.conv1.out', '1.conv2.in', '1.conv2.out'])
        [self.assertEqual(len(v), self.n) for v in y.values()]
        y = evaluate_intermediates(self.model, self.dataloader, device="cpu", exclude=["out"])
        npt.assert_array_equal(list(y.keys()), ['0.in', '1.conv1.in', '1.bn1.in',
            '1.relu1.in', '1.conv2.in', '1.bn2.in', '1.shortcut.1.in', '1.relu2.in'])
        [self.assertEqual(len(v), self.n) for v in y.values()]

    def test_eval_model(self):
        y = evaluate_model(self.model, self.dataloader, device="cpu")
        self.assertEqual(len(y), self.n)
        npt.assert_array_almost_equal(y, self.model(self.data).detach().numpy())
        y = evaluate_model(self.model, self.dataloader, device="cpu")

    def test_prediction_depth(self):
        y = evaluate_intermediates(self.model, self.dataloader, device="cpu")
        pd = prediction_depth(y, self.labels, k=2)
        self.assertEqual(pd.shape, (self.n,))
        self.assertTrue(np.all(0 <= pd))
        self.assertTrue(np.all(pd <= len(y)))


if __name__ == '__main__':
    unittest.main()
