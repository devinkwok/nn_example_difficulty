import unittest
from collections import defaultdict
import numpy.testing as npt
import torch

from difficulty.test.base import BaseTest
from difficulty.model.eval import evaluate_model, batch_eval_intermediates, evaluate_intermediates, find_intermediate_layers, combine_batches, split_batches
from difficulty.metrics import *


class TestModel(BaseTest):

    def setUp(self):
        super().setUp()
        self.data_shape = self.data[0].shape
        self.layers = find_intermediate_layers(self.model, self.data_shape, device=self.device, include=["relu"])
        self.all_layers = find_intermediate_layers(self.model, self.data_shape, device=self.device)

    def test_find_intermediate_layers(self):
        npt.assert_array_equal(self.all_layers,
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
        # only get input/output to top level module
        first_module, *_  = self.model.named_modules()
        layers = find_intermediate_layers(self.model, self.data_shape, device=self.device, named_modules=[first_module])
        npt.assert_array_equal(layers, ['.in', '.out'])
        # include selected layers
        layers = find_intermediate_layers(self.model, self.data_shape, device=self.device, include=["fc"])
        npt.assert_array_equal(layers, ['fc.in', 'fc.out'])
        # exclude selected layers
        layers = find_intermediate_layers(self.model, self.data_shape, device=self.device, exclude=["conv", "bn", "relu", "shortcut", ".in", "fc."])
        npt.assert_array_equal(layers, ['blocks.0.out', 'blocks.1.out', 'blocks.2.out', 'blocks.3.out', 'blocks.4.out', 'blocks.5.out', ".out"])

    def _intermediates_generator(self, layers):  # wrapper to fill in some arguments
        return batch_eval_intermediates(self.model, self.dataloader, layers, device=self.device)

    def _evaluate_intermediates(self, layers):  # wrapper to fill in some arguments
        return evaluate_intermediates(self.model, self.dataloader, layers, device=self.device)

    def test_evaluate_intermediates(self):
        with torch.no_grad():
            layers_subset = ['blocks.0.out', 'blocks.1.out', 'blocks.2.out', 'blocks.3.out', 'blocks.4.out', 'blocks.5.out', ".out"]
            # include layers_subset because they are duplicates in self.all_layers, and thus removed by find_intermediate_layers
            # this allows comparison with evaluating only layers_subset later
            input, hidden, output, labels = self._evaluate_intermediates(self.all_layers + layers_subset)
            self.tensors_equal(input, self.data)
            self.all_close(output, self.model(self.data.to(device=self.device)))
            self.tensors_equal(labels, self.data_labels)
            layers = list(hidden.keys())
            self.tensors_equal(hidden[layers[0]], input)
            self.tensors_equal(hidden[layers[-1]], output)
            [self.assertEqual(len(v), self.n) for v in hidden.values()]
            # intermediates for only selected layers
            _, hidden_2, _, _ = self._evaluate_intermediates(layers_subset)
            self.dict_all_close({k: v for k, v in hidden.items() if k in layers_subset}, hidden_2)

    def test_eval_model(self):
        y, labels, acc, loss = evaluate_model(self.model, self.dataloader, device=self.device)
        self.assertEqual(len(y), self.n)
        self.all_close(y, self.model(self.data.to(device=self.device)))
        self.tensors_equal(labels, self.data_labels)
        self.assertIsNone(acc)
        self.assertIsNone(loss)
        y, labels, acc, loss = evaluate_model(self.model, self.dataloader, device=self.device,
                                              return_accuracy=True, loss_fn=lambda x, y: torch.argmax(x, dim=-1) == y)
        self.tensors_equal(acc, self.data_labels == torch.argmax(y.detach().cpu(), dim=-1))
        self.all_close(loss, acc)

    def test_combine_batches(self):
        z = self._intermediates_generator(self.layers)
        inputs, hidden, outputs, labels = combine_batches(z, len(self.data))
        self.dict_all_close(hidden, self._evaluate_intermediates(self.layers)[1])
        gen = split_batches(inputs, hidden, outputs, labels, self.batch_size)
        for a, b in zip(gen, self._intermediates_generator(self.layers)):
            self.tensors_equal(a[0], b[0])  # inputs
            self.dict_all_close(a[1], b[1]) # intermediates
            self.all_close(a[2], b[2])      # outputs
            self.tensors_equal(a[3], b[3])  # labels

    def test_prediction_depth(self):
        # check that outputs are correct shape and ranges
        _, z, _, _ = self._evaluate_intermediates(self.layers)
        train_z = [v[:self.n // 2] for v in z.values()]
        train_labels = self.data_labels[:self.n // 2]
        pd = prediction_depth(train_z, train_labels, z, self.data_labels, k=2)
        self.assertEqual(pd.shape, (self.n,))
        self.assertTrue(torch.all(0 <= pd))
        self.assertTrue(torch.all(pd <= len(z)))

        # check that classifying the training points with k=1 is always identical to training labels
        pd = prediction_depth(z, self.data_labels, z, self.data_labels, k=1)
        self.assertEqual(pd.shape, (self.n,))
        self.tensors_equal(pd, torch.zeros_like(pd))  # always correct
        # flipping labels makes prediction depth always the max
        pd = prediction_depth(z, 1 - self.data_labels, z, self.data_labels, k=1)
        self.tensors_equal(pd, torch.full_like(pd, len(z)))
        # check that dict input works, and omitting test points classifies training points
        pd = prediction_depth(z, self.data_labels, k=1)
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

    def test_intermediates_iterable(self):
        # check that using a generator to get intermediates for prediction_depth is the same as passing a dict of intermediates
        # train only
        generator = intermediates_iterable(self.model, self.dataloader, self.layers, device=self.device)
        pd = prediction_depth(generator, self.data_labels, k=3)
        _, z, _, _ = self._evaluate_intermediates(self.layers)
        self.tensors_equal(pd, prediction_depth(z, self.data_labels, k=3))
        # train and test
        train_generator = intermediates_iterable(self.model, self.dataloader, self.layers, device=self.device)
        test_generator = intermediates_iterable(self.model, self.dataloader, self.layers, device=self.device)
        pd = prediction_depth(train_generator, 1 - self.data_labels, test_generator, self.data_labels, k=4)
        _, z, _, _ = self._evaluate_intermediates(self.layers)
        self.tensors_equal(pd, prediction_depth(z, 1 - self.data_labels, z, self.data_labels, k=4))

    def test_prototypes(self):
        repr_layer = 'blocks.5.relu2.out'
        _, z, _, _ = evaluate_intermediates(self.model, self.dataloader, [repr_layer], device=self.device)
        distances = supervised_prototypes(z[repr_layer], self.data_labels)
        self.assertEqual(distances.shape, (self.n,))
        distances, kmeans = self_supervised_prototypes(z[repr_layer], k=10, return_kmeans_obj=True)
        self.assertEqual(distances.shape, (self.n,))
        # supervised using labels from self-supervised should be equal
        dist_supervised = supervised_prototypes(z[repr_layer], torch.tensor(kmeans.labels_))
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
        _, z, outputs, labels = evaluate_intermediates(self.model, self.dataloader, self.all_layers, device=self.device)
        repr = list(z.values())[-1]

        # use default values
        pd = prediction_depth(z, self.data_labels)
        proto = supervised_prototypes(repr.to(dtype=torch.float64), self.data_labels)
        selfproto = self_supervised_prototypes(repr.to(dtype=torch.float64), random_state=SEED)
        scores = representation_metrics(self.model, self.dataloader, device=self.device, selfproto_random_state=SEED, verbose=True)
        self.all_close(scores["pd"], pd)
        self.all_close(scores["proto"], proto)
        self.all_close(scores["selfproto"], selfproto)

        # use non-defaults
        proto_layer = 'fc.in'
        representations = z[proto_layer]
        test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.data, 1 - self.data_labels))
        _, z, outputs, labels = evaluate_intermediates(self.model, self.dataloader, self.layers, device=self.device)
        z_softmax = list(z.values()) + [torch.nn.functional.softmax(repr, dim=-1)]
        pd = prediction_depth(z_softmax, self.data_labels, z_softmax, 1 - self.data_labels, k=2)
        proto = supervised_prototypes(representations.to(dtype=torch.float64), self.data_labels)
        selfproto = self_supervised_prototypes(representations.to(dtype=torch.float64), k=10, random_state=SEED)
        scores = representation_metrics(self.model, self.dataloader, device=self.device, generate_pointwise_metrics=True,
                                        pd_layers=self.layers, pd_append_softmax=True,
                                        pd_test_dataloader=test_dataloader, pd_k=2, pd_return_layerpred=True,
                                        proto_layer=proto_layer, selfproto_k=10, selfproto_random_state=SEED)
        self.tensors_equal(scores["pd"], pd)
        self.all_close(scores["proto"], proto)
        self.all_close(scores["selfproto"], selfproto)
        for k, v in pointwise_metrics(outputs, labels).items():
            self.all_close(scores[k], v)

        layers = [scores[f"pdlayer{i}"] for i in range(len(self.layers) + 1)]
        self.tensors_equal(first_unforgettable(torch.stack(layers, dim=0)), scores["pd"])


if __name__ == '__main__':
    unittest.main()
