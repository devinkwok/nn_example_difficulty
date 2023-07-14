import torch
import unittest
from functools import partial
import numpy.testing as npt

from difficulty.test.base import BaseTest
from difficulty.metrics import *
from difficulty.metrics.accumulator import BatchAccumulator


class TestMetrics(BaseTest):

    def setUp(self) -> None:
        super().setUp()
        self.test_data = {
            "bool": torch.randn(2, 3, 2) > 0.,
            "randn": torch.randn(10, 20, 100, dtype=torch.float32),
            "arange-large": torch.arange(100*1000*3, dtype=torch.float32).reshape(100, 1000, 3),
            "randn-high-bias": torch.randn(100, 2, 2) + torch.full((100, 2, 2), 100),
            "randn-high-var": torch.cat([torch.randn(100, 2, 2), torch.full((2, 2, 2), 100)], dim=0),
        }
        self.dtype = torch.float64

    def _test_accumulator(self, AccumulateClass, data, identity_value, ref_fn):
        # identity AxBxC
        with self.ArgsUnchanged(data):
            obj = AccumulateClass()
            obj.add(data)
            self.all_close(obj.get(), identity_value)
        # compute over A elements of size BxC
        with self.ArgsUnchanged(data):
            obj = AccumulateClass()
            for i, y in enumerate(data):
                obj.add(y)
            self.all_close(obj.get(), ref_fn(data, dim=0))
        # compute over AxB elements of size C
        obj, obj_T = AccumulateClass(), AccumulateClass()
        for y in data:
            with self.ArgsUnchanged(data, y):
                obj.add(y, dim=0)
                obj_T = obj_T.add(y.T, dim=1)
        self.all_close(obj.get(), ref_fn(data.reshape(-1, data.shape[-1]), dim=0))
        self.all_close(obj.get(), obj_T.get())
        # save and load, and also check return value of add()
        with self.ArgsUnchanged(data):
            obj = AccumulateClass()
            for y in data:
                obj = obj.add(y, dim=0)
                obj.save(self.tmp_file)
                obj = obj.load(self.tmp_file)
            self.all_close(obj.get(), obj_T.get())

    def test_mean(self):
        mean_fn = lambda x, dim: torch.mean(x.to(dtype=self.dtype), dim=dim)
        for msg, data in self.test_data.items():
            with self.subTest(msg, data=data[0, 0, 0]):
                self._test_accumulator(OnlineMean,
                                       data,
                                       data.to(self.dtype),
                                       mean_fn)
        self.check_accumulator_metadata(OnlineMean, data)

    def test_variance(self):
        var_fn = lambda x, dim: torch.var(x.to(dtype=self.dtype), dim=dim)
        for msg, data in self.test_data.items():
            with self.subTest(msg, data=data[0, 0, 0]):
                self._test_accumulator(OnlineVariance,
                                       data,
                                       torch.full_like(data, torch.nan, dtype=self.dtype),
                                       var_fn)
        self.check_accumulator_metadata(OnlineMean, data)

    def test_batch_accumulator(self):
        n_batches = 3
        for msg, data in self.test_data.items():
            torch.moveaxis(data, 0, -1)  # batch accumulator works on last axis
            # check that subset does not modify data if no minibatch supplied
            obj = BatchAccumulator(data.shape[-1], dtype=data.dtype)
            subset = obj.add(data)
            npt.assert_array_equal(subset, data)
            npt.assert_array_equal(obj.n, 1)
            if data.shape[-1] < n_batches:
                continue
            idx = torch.randperm(data.shape[-1])
            output = torch.zeros_like(data)
            for i in range(1, 3):
                for subset_idx in torch.split(idx, len(idx) // n_batches):
                    # check that subset works
                    sub_in, sub_out = obj.add(data, output, minibatch_idx=subset_idx)
                    npt.assert_array_equal(sub_in, data[..., subset_idx])
                    npt.assert_array_equal(sub_out, (i - 1) * data[..., subset_idx])
                    # check that update works
                    obj.update_subset_(output, sub_in + sub_out, subset_idx)
                    npt.assert_array_equal(output[..., subset_idx], i * data[..., subset_idx])
                    # check that n is updated correctly
                    npt.assert_array_equal(obj.n[..., subset_idx], i + 1)
                npt.assert_array_equal(output, i * data)
                npt.assert_array_equal(obj.n, i + 1)
        # check that metadata is preserved in save/load
        self.check_accumulator_metadata(partial(BatchAccumulator, data.shape[-1]), data)


if __name__ == '__main__':
    unittest.main()
