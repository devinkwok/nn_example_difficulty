import os
import unittest
import warnings
from typing import List
from pathlib import Path
import numpy as np
import numpy.testing as npt
import torch

from difficulty.metrics import zero_one_accuracy
from difficulty.test.example_model import Model


class BaseTest(unittest.TestCase):

    class ArgsUnchanged:
        """Context manager for testing that arguments are not modified in place
        by some operation.
        """
        def __init__(self, *args: List[np.ndarray]) -> None:
            self.references = args
            self.originals = [np.copy(x) for x in args]

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_value, exc_tb):
            for x, y in zip(self.references, self.originals):
                npt.assert_array_equal(x, y)

    def setUp(self):
        N = 20
        C = 10
        I = 15
        S = 5
        R = 3
        # useful variables for pointwise metrics
        self.n_examples = N
        self.n_class = C
        self.n_steps = I
        self.logits = [
            torch.randn(I, N, C),
            torch.randn(R, I, N, C),
            torch.randn(R, I, S, N, C),
        ]
        self.logit_labels = [
            torch.randint(0, C, (I, N)),
            torch.randint(0, C, (R, I, N)),
            torch.randint(0, C, (R, I, S, N)),
        ]
        self.zero_labels = [
            torch.zeros([I, N], dtype=int),
            torch.zeros([R, I, N], dtype=int),
            torch.zeros([R, I, S, N], dtype=int),
        ]
        self.acc = [zero_one_accuracy(x, y) for x, y in zip(self.logits, self.logit_labels)]

        self.zeros = torch.zeros([I, N])
        self.ones = torch.ones([I, N])
        self.zero_to_ones = torch.cat([self.zeros[..., 0:1, :], self.ones[..., 1:, :]], axis=-2)
        self.one_to_zeros = 1 - self.zero_to_ones

        # useful variables for gradients and accumulators
        self.n = 40*2
        self.batch_size = 15
        self.data = torch.randn([self.n, 3, 11, 9])
        self.data_labels = torch.cat([torch.zeros(self.n // 2),
                                      torch.ones(self.n - self.n // 2)]).to(dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(self.data, self.data_labels)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, drop_last=False)
        self.model = Model.create()
        self.tmp_file = Path(f"difficulty/test/TEMP_TEST_DATA/{type(self).__name__}_save_file.npz")
        self.tmp_file.parent.mkdir(parents=True, exist_ok=True)
        self.n_inputs = torch.prod(torch.tensor(self.data.shape[1:]))
        self.n_outputs = 10
        self.epsilon = 1e-6

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            warnings.warn("CUDA device not available, only testing on cpu.")
            self.device = torch.device("cpu")

    def tearDown(self) -> None:
        if self.tmp_file.exists():
            os.remove(self.tmp_file)

    def check_accumulator_metadata(self, AccumulateClass, data):
        metadata = {"meta_str": "string", "data_int": -99, "list_float": 0.1234}
        # test save/load metadata at init
        obj = AccumulateClass(**metadata)
        obj.save(self.tmp_file)
        obj = obj.load(self.tmp_file)
        {self.assertEqual(obj.metadata[k], v) for k, v in metadata.items()}
        # test save/load list metadata at add
        for i, y in enumerate(data):
            obj.add(y, **metadata, count=i)
            obj.save(self.tmp_file)
            obj = obj.load(self.tmp_file)
            {self.assertListEqual(obj.metadata_lists[k], [v]*(i+1))
                for k, v in metadata.items()}
            self.assertListEqual(obj.metadata_lists["count"], list(range(i+1)))

    def tensors_equal(self, X, Y):
        npt.assert_array_equal(X.detach().cpu().numpy(), Y.detach().cpu().numpy())

    def all_close(self, X, Y):
        # this test is more forgiving to account for gpu noise and low precision
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu()
        if isinstance(Y, torch.Tensor):
            Y = Y.detach().cpu()
        torch.testing.assert_close(X + self.epsilon,
                                   Y + self.epsilon,
                                   atol=1e-5, rtol=1e-4, equal_nan=True, check_dtype=False)

    def dict_all_close(self, X, Y):
        npt.assert_array_equal(sorted(list(X.keys())), sorted(set(Y.keys())))
        for k, v in X.items():
            self.all_close(v, Y[k])
