import unittest
from collections import defaultdict
import numpy.testing as npt
import torch

from open_lth.models import cifar_resnet
from open_lth.models.initializers import kaiming_normal
from difficulty.model.eval import evaluate_model, evaluate_intermediates
from difficulty.metrics import *


class TestModel(unittest.TestCase):

    def setUp(self):
        self.n = 5*2
        self.batch_size = 4
        self.data = torch.randn([self.n, 3, 9, 9])
        self.labels = torch.cat([torch.zeros(self.n // 2), torch.ones(self.n - self.n // 2)])
        dataset = torch.utils.data.TensorDataset(self.data, self.labels)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, drop_last=False)
        self.model = cifar_resnet.Model.get_model_from_name("cifar_resnet_14_8", initializer=kaiming_normal)

    def _combine_batches(self, generator):
        inputs, outputs, labels = [], [], []
        hiddens = defaultdict(list)
        for input, hidden, output, label in generator:
            inputs.append(input)
            outputs.append(output)
            labels.append(label)
            for k, v in hidden.items():
                hiddens[k].append(v)
        for k, v in hiddens.items():
            hiddens[k] = torch.cat(v, dim=0)
        return torch.cat(inputs, dim=0), hiddens, torch.cat(outputs, dim=0), torch.cat(labels, dim=0)

    def test_evaluate_intermediates(self):
        with torch.no_grad():
            y = evaluate_intermediates(self.model, self.dataloader, device="cpu")
            input, hidden, output, labels = self._combine_batches(y)
            npt.assert_array_equal(input, self.data)
            npt.assert_array_equal(output, self.model(self.data))
            npt.assert_array_equal(labels, self.labels)
            layers = list(hidden.keys())
            npt.assert_array_equal(hidden[layers[0]], input)
            npt.assert_array_equal(hidden[layers[-1]], output)
            [self.assertEqual(len(v), self.n) for v in hidden.values()]
            npt.assert_array_equal(list(hidden.keys()),
                ['conv.in',                 # input
                 'conv.out',
                 'bn.out',
                 'relu.out',                # block 0 input
                 'blocks.0.conv1.out',
                 'blocks.0.bn1.out',
                 'blocks.0.relu1.out',
                 'blocks.0.conv2.out',
                 'blocks.0.bn2.out',
                 'blocks.0.relu2.in',       # skip connection
                 'blocks.0.relu2.out',      # block 1 input
                 'blocks.1.conv1.out',
                 'blocks.1.bn1.out',
                 'blocks.1.relu1.out',
                 'blocks.1.conv2.out',
                 'blocks.1.bn2.out',
                 'blocks.1.relu2.in',       # skip connection
                 'blocks.1.relu2.out',      # block 2 input
                 'blocks.2.conv1.out',
                 'blocks.2.bn1.out',
                 'blocks.2.relu1.out',
                 'blocks.2.conv2.out',
                 'blocks.2.bn2.out',
                 'blocks.2.shortcut.0.out', # conv on skip path
                 'blocks.2.shortcut.1.out', # bn on skip path
                 'blocks.2.relu2.in',       # skip connection
                 'blocks.2.relu2.out',      # block 3 input
                 'blocks.3.conv1.out',
                 'blocks.3.bn1.out',
                 'blocks.3.relu1.out',
                 'blocks.3.conv2.out',
                 'blocks.3.bn2.out',
                 'blocks.3.relu2.in',       # skip connection
                 'blocks.3.relu2.out',      # block 4 input
                 'blocks.4.conv1.out',
                 'blocks.4.bn1.out',
                 'blocks.4.relu1.out',
                 'blocks.4.conv2.out',
                 'blocks.4.bn2.out',
                 'blocks.4.shortcut.0.out', # conv on skip path
                 'blocks.4.shortcut.1.out', # bn on skip path
                 'blocks.4.relu2.in',       # skip connection
                 'blocks.4.relu2.out',      # block 5 input
                 'blocks.5.conv1.out',
                 'blocks.5.bn1.out',
                 'blocks.5.relu1.out',
                 'blocks.5.conv2.out',
                 'blocks.5.bn2.out',
                 'blocks.5.relu2.in',       # skip connection
                 'blocks.5.relu2.out',      # linear classifier input
                 'fc.in',                   # pooling
                 'fc.out'                   # output
                 ])
            # only save input/output to top level module
            first_module, *_  = self.model.named_modules()
            y = evaluate_intermediates(self.model, self.dataloader, device="cpu", named_modules=[first_module])
            _, hidden, _, _ = self._combine_batches(y)
            npt.assert_array_equal(hidden['.in'], input)
            npt.assert_array_equal(hidden['.out'], output)
            npt.assert_array_equal(list(hidden.keys()), ['.in', '.out'])
            # include selected layers
            y = evaluate_intermediates(self.model, self.dataloader, device="cpu", include=["fc"])
            _, hidden, _, _ = self._combine_batches(y)
            npt.assert_array_equal(list(hidden.keys()), ['fc.in', 'fc.out'])
            # exclude selected layers
            y = evaluate_intermediates(self.model, self.dataloader, device="cpu", exclude=["conv", "bn", "relu", "shortcut", ".in", "fc."])
            _, hidden, _, _ = self._combine_batches(y)
            npt.assert_array_equal(list(hidden.keys()), ['blocks.0.out', 'blocks.1.out', 'blocks.2.out', 'blocks.3.out', 'blocks.4.out', 'blocks.5.out', ".out"])

    def test_eval_model(self):
        y = evaluate_model(self.model, self.dataloader, device="cpu")
        self.assertEqual(len(y), self.n)
        npt.assert_array_almost_equal(y, self.model(self.data).detach().numpy())
        y = evaluate_model(self.model, self.dataloader, device="cpu")

    def test_prediction_depth(self):
        _, y, _, _ = self._combine_batches(evaluate_intermediates(self.model, self.dataloader, device="cpu"))
        pd = prediction_depth(y, self.labels, k=2)
        self.assertEqual(pd.shape, (self.n,))
        self.assertTrue(torch.all(0 <= pd))
        self.assertTrue(torch.all(pd <= len(y)))


if __name__ == '__main__':
    unittest.main()
