import os
import torch
import unittest
from pathlib import Path

from difficulty.metrics import accumulator
from difficulty.test.test_metrics import ArgsUnchanged


class TestMetrics(unittest.TestCase):

    def setUp(self) -> None:
        self.data = {
            "randn-small": torch.randn(2, 3, 2),
            "randn-large": torch.randn(10, 20, 100, dtype=torch.float32),
            "arange-small": torch.arange(8).reshape(2, 2, 2).to(dtype=torch.float32),
            "arange-large": torch.arange(6*4*3, dtype=torch.float32).reshape(6, 4, 3),
        }
        self.tmp_file = Path("difficulty/test/tmp_test_accumulator_save_file.npz")

    def tearDown(self) -> None:
        os.remove(self.tmp_file)

    @staticmethod
    def assert_tensor_equal(x, y, rtol=0.00001, atol=1e-8):
        if x.shape != y.shape:
            raise AssertionError(x.shape, y.shape)
        if not torch.all(torch.isnan(x) == torch.isnan(y)):
            if not torch.allclose(x, y, rtol=rtol, atol=atol):
                idx = torch.where(torch.abs(y - x) > atol)
                raise AssertionError(x[idx], y[idx])
        return True

    def _test_accumulator(self, AccumulateClass, ref_fn, data, identity_value):
        # identity AxBxC
        with ArgsUnchanged(data):
            obj = AccumulateClass()
            obj = obj.add(data)
            self.assert_tensor_equal(obj.get(), identity_value)
        # compute over A elements of size BxC
        with ArgsUnchanged(data):
            obj = AccumulateClass()
            for y in data:
                obj = obj.add(y)
            self.assert_tensor_equal(obj.get(), ref_fn(data, dim=0))
        # compute over AxB elements of size C
        with ArgsUnchanged(data):
            obj, obj_T = AccumulateClass(), AccumulateClass()
            for y in data:
                obj = obj.add(y, dim=0)
                obj_T = obj_T.add(y.T, dim=1)
            self.assert_tensor_equal(obj.get(), ref_fn(data.reshape(-1, data.shape[-1]), dim=0))
            self.assert_tensor_equal(obj.get(), obj_T.get())
        # save and load
        with ArgsUnchanged(data):
            obj = AccumulateClass()
            for y in data:
                obj = obj.add(y, dim=0)
                obj.save(self.tmp_file)
                obj = obj.load(self.tmp_file)
            self.assert_tensor_equal(obj.get(), obj_T.get())

    def test_mean(self):
        for msg, data in self.data.items():
            with self.subTest(msg, data=data[0, 0, 0]):
                self._test_accumulator(accumulator.OnlineMean, torch.mean, data, data)

    def test_variance(self):
        for msg, data in self.data.items():
            with self.subTest(msg, data=data[0, 0, 0]):
                self._test_accumulator(accumulator.OnlineVariance, lambda x, dim: torch.std(x, dim=dim)**2, data, torch.full_like(data, torch.nan))

if __name__ == '__main__':
    unittest.main()


