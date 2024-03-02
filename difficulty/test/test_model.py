import unittest
from collections import defaultdict
import numpy.testing as npt
import torch

from difficulty.test.base import BaseTest
from difficulty.model.eval import evaluate_model, evaluate_intermediates, combine_batches
from difficulty.metrics import *


class TestModel(BaseTest):

    def test_evaluate_intermediates(self):
        with torch.no_grad():
            y = evaluate_intermediates(self.model, self.dataloader, device=self.device)
            input, hidden, output, labels = combine_batches(y)
            self.tensors_equal(input, self.data)
            self.all_close(output, self.model(self.data.to(device=self.device)))
            self.tensors_equal(labels, self.data_labels)
            layers = list(hidden.keys())
            self.tensors_equal(hidden[layers[0]], input)
            self.tensors_equal(hidden[layers[-1]], output)
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
            y = evaluate_intermediates(self.model, self.dataloader, device=self.device, named_modules=[first_module])
            _, hidden, _, _ = combine_batches(y)
            self.tensors_equal(hidden['.in'], input)
            self.tensors_equal(hidden['.out'], output)
            npt.assert_array_equal(list(hidden.keys()), ['.in', '.out'])
            # include selected layers
            y = evaluate_intermediates(self.model, self.dataloader, device=self.device, include=["fc"])
            _, hidden, _, _ = combine_batches(y)
            npt.assert_array_equal(list(hidden.keys()), ['fc.in', 'fc.out'])
            # exclude selected layers
            y = evaluate_intermediates(self.model, self.dataloader, device=self.device, exclude=["conv", "bn", "relu", "shortcut", ".in", "fc."])
            _, hidden, _, _ = combine_batches(y)
            npt.assert_array_equal(list(hidden.keys()), ['blocks.0.out', 'blocks.1.out', 'blocks.2.out', 'blocks.3.out', 'blocks.4.out', 'blocks.5.out', ".out"])

    def test_eval_model(self):
        y, _, _, _ = evaluate_model(self.model, self.dataloader, device=self.device)
        self.assertEqual(len(y), self.n)
        self.all_close(y, self.model(self.data.to(device=self.device)))

    def subset_pd_intermediates(self, intermediates):
        return [v for k, v in intermediates.items() if "relu" in k]

    def test_prediction_depth(self):
        # check that outputs are correct shape and ranges
        _, y, out, _ = combine_batches(evaluate_intermediates(self.model, self.dataloader, device=self.device))
        intermediates = self.subset_pd_intermediates(y)
        train_data = [v[:self.n // 2] for v in intermediates]
        train_labels = self.data_labels[:self.n // 2]
        train_out = out[:self.n // 2]
        pd = prediction_depth(train_data, train_labels, intermediates, self.data_labels, k=2)
        self.assertEqual(pd.shape, (self.n,))
        self.assertTrue(torch.all(0 <= pd))
        self.assertTrue(torch.all(pd <= len(intermediates)))

        # check that object is the same as function
        pd_obj = PredictionDepth(train_data, train_labels, k=2)
        obj_outputs = []
        # run over batches
        for _, x, _, labels in evaluate_intermediates(self.model, self.dataloader, device=self.device):
            x = self.subset_pd_intermediates(x)
            obj_outputs.append(pd_obj(x, labels))
        self.all_close(torch.cat(obj_outputs, dim=0), pd)

        # check that softmax of outputs is appended
        pd = prediction_depth(train_data, train_labels, intermediates, self.data_labels,
                              train_outputs=train_out, test_outputs=out, k=2)
        self.assertTrue(torch.any(pd > len(intermediates)))

        # check that classifying the training points with k=1 is always identical to training labels
        pd = prediction_depth(intermediates, self.data_labels, intermediates, self.data_labels, k=1)
        self.assertEqual(pd.shape, (self.n,))
        self.tensors_equal(pd, torch.zeros_like(pd))  # always correct
        # flipping labels makes prediction depth always the max
        pd = prediction_depth(intermediates, 1 - self.data_labels, intermediates, self.data_labels, k=1)
        self.tensors_equal(pd, torch.full_like(pd, len(intermediates)))
        # check that dict input works, and omitting test points classifies training points
        pd = prediction_depth(y, self.data_labels, k=1)
        self.tensors_equal(pd, torch.zeros_like(pd))

        # check classification in an artificial task
        # train points are fixed at 0 and 1 in all layers
        train_points = torch.stack([torch.zeros(3), torch.ones(3)], dim=1)
        train_labels = torch.tensor([0, 1])
        # examples start from 0.2n and shift by 0.2 per layer, prediction changes at threshold 0.5
        test_points = torch.linspace(0, 0.6, 4).reshape(-1, 1)
        labels = torch.ones(4)
        right_shift = [test_points + i*0.2 for i in range(3)]
        pd = prediction_depth(train_points, train_labels, right_shift, labels, k=1)
        self.tensors_equal(pd, torch.tensor([3, 2, 1, 0]))

    def test_prototypes(self):
        _, y, _, _ = combine_batches(evaluate_intermediates(
            self.model, self.dataloader, device=self.device, include=['blocks.5.relu2.']))
        distances = supervised_prototypes(y['blocks.5.relu2.out'], self.data_labels)
        self.assertEqual(distances.shape, (self.n,))
        distances, kmeans = self_supervised_prototypes(y['blocks.5.relu2.out'], k=10, return_kmeans_obj=True)
        self.assertEqual(distances.shape, (self.n,))
        # supervised using labels from self-supervised should be equal
        dist_supervised = supervised_prototypes(y['blocks.5.relu2.out'], torch.tensor(kmeans.labels_))
        self.all_close(distances, dist_supervised)

        # artificial data: clusters x and y with differing means
        x = torch.randn(5000, 50)
        y = torch.randn(5000, 50) + torch.ones(50).reshape(1, -1)
        x_mean = torch.mean(x, dim=0)
        y_mean = torch.mean(y, dim=0)
        representations = torch.cat([x, y], dim=0)
        labels = torch.tensor([0]*5000 + [1]*5000)

        # distances in supervised_prototypes should be somewhat less (due to empirical mean) than std, which is sqrt(n)
        distance = supervised_prototypes(representations, labels)
        mean_diff = (torch.sqrt(torch.tensor(50)) - torch.mean(distance)).item()
        self.assertTrue(mean_diff > 0 and mean_diff < 1e-1)
        # check that distance is calculated from empirical mean of each class
        means = torch.cat([x_mean.broadcast_to(5000, 50), y_mean.broadcast_to(5000, 50)], dim=0)
        dist_from_means = torch.linalg.norm(representations - means, ord=2, dim=-1)
        self.all_close(distance, dist_from_means)

        # distances in self_supervised_prototypes with k=2 should be close to supervised with high probability
        dist_self, kmeans = self_supervised_prototypes(representations, k=2, return_kmeans_obj=True)
        n_distance_diff = torch.count_nonzero(torch.abs(dist_self - distance) > 1e-2)
        self.assertLess(n_distance_diff, 0.01 * len(labels))
        # misclassified objects have less distance with high probability
        agreement = torch.tensor(kmeans.labels_) == labels
        n = torch.count_nonzero(agreement)
        misclassified = torch.logical_not(agreement) if n > len(agreement) - n else agreement
        self.assertLess(torch.count_nonzero(dist_self[misclassified] > dist_from_means[misclassified]), 2)


    def test_representation_metrics(self):
        SEED = 42
        _, y, outputs, labels = combine_batches(evaluate_intermediates(self.model, self.dataloader, device=self.device))

        # use default values
        pd = prediction_depth(y, self.data_labels)
        representation = list(y.values())[-1]
        proto = supervised_prototypes(representation, self.data_labels)
        selfproto = self_supervised_prototypes(representation, random_state=SEED)
        scores = representation_metrics(self.model, self.dataloader, device=self.device, selfproto_random_state=SEED)
        self.all_close(scores["pd"], pd)
        self.all_close(scores["proto"], proto)
        self.all_close(scores["selfproto"], selfproto)

        # use non-defaults
        test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.data, 1 - self.data_labels))
        y_softmax = list(y.values()) + [torch.nn.functional.softmax(representation, dim=-1)]
        pd = prediction_depth(y_softmax, self.data_labels, y_softmax, 1 - self.data_labels, k=2)
        proto = supervised_prototypes(y['blocks.5.relu2.out'], self.data_labels)
        selfproto = self_supervised_prototypes(y['blocks.5.relu2.out'], k=10, random_state=SEED)
        scores = representation_metrics(self.model, self.dataloader, device=self.device, generate_pointwise_metrics=True,
                                        pd_append_softmax=True, pd_test_dataloader=test_dataloader, pd_k=2,
                                        proto_layer='blocks.5.relu2.out', selfproto_k=10, selfproto_random_state=SEED)
        self.all_close(scores["pd"], pd)
        self.all_close(scores["proto"], proto)
        self.all_close(scores["selfproto"], selfproto)
        for k, v in pointwise_metrics(outputs, labels).items():
            self.all_close(scores[k], v)


if __name__ == '__main__':
    unittest.main()
