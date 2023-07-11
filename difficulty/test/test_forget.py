import unittest
import numpy.testing as npt

from difficulty.metrics import *
from difficulty.metrics.forget import _forgetting_events, _learning_events
from difficulty.test.utils import ArgsUnchanged


class TestMetrics(unittest.TestCase):

    def setUp(self):
        N = 20
        C = 10
        I = 15
        S = 5
        R = 3
        self.n_examples = N
        self.n_class = C
        self.n_steps = I
        self.logits = [
            torch.randn(I, N, C),
            torch.randn(R, I, N, C),
            torch.randn(R, I, S, N, C),
        ]
        self.labels = [
            torch.randint(0, C, (I, N)),
            torch.randint(0, C, (R, I, N)),
            torch.randint(0, C, (R, I, S, N)),
        ]
        self.zero_labels = [
            torch.zeros([I, N], dtype=int),
            torch.zeros([R, I, N], dtype=int),
            torch.zeros([R, I, S, N], dtype=int),
        ]
        self.acc = [zero_one_accuracy(x, y) for x, y in zip(self.logits, self.labels)]

        self.zeros = torch.zeros([I, N])
        self.ones = torch.ones([I, N])
        self.zero_to_ones = torch.cat([self.zeros[..., 0:1, :], self.ones[..., 1:, :]], axis=-2)
        self.one_to_zeros = 1 - self.zero_to_ones
        self.checkerboard = torch.tensor([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0]], dtype=bool)

    def _common_tests(self, apply_fn, is_order_statistic=True):
        for acc in self.acc:
            with ArgsUnchanged(acc):
                x = apply_fn(acc)
            if is_order_statistic:
                self.assertEqual(x.shape, acc.shape[:-2] + acc.shape[-1:])
                npt.assert_array_less(x, acc.shape[-2] + 1)
            else:
                self.assertEqual(x.shape, acc.shape)
                transposed = acc.moveaxis(-2, 0)
                npt.assert_array_equal(x.moveaxis(-2, 0), apply_fn(transposed, dim=0))

    def test_forgetting_events(self):
        self._common_tests(_forgetting_events, is_order_statistic=False)
        self._common_tests(lambda x, dim=-2: _forgetting_events(
            x, start_at_zero=False, dim=dim), is_order_statistic=False)
        # forgetting from trained: 1 event at start if all zeros
        npt.assert_array_equal(_forgetting_events(self.zeros, start_at_zero=False)[..., 0, :], 1)
        npt.assert_array_equal(_forgetting_events(self.zeros, start_at_zero=False)[..., 1:, :], 0)
        # all other cases of all ones/zeros: no events
        npt.assert_array_equal(_forgetting_events(self.ones, start_at_zero=False), 0)
        npt.assert_array_equal(_forgetting_events(self.ones, start_at_zero=True), 0)
        npt.assert_array_equal(_forgetting_events(self.zeros, start_at_zero=True), 0)
        # checkerboard case
        npt.assert_array_equal(_forgetting_events(self.checkerboard, dim=-1, start_at_zero=False), torch.logical_not(self.checkerboard))

    def test_learning_events(self):
        self._common_tests(_learning_events, is_order_statistic=False)
        # check that learning does not overlap with forgetting
        for acc in self.acc:
            with ArgsUnchanged(acc):
                self.assertFalse(torch.any(torch.logical_and(
                    _learning_events(acc), _forgetting_events(acc))))
                self.assertFalse(torch.any(torch.logical_and(
                    _learning_events(acc, start_at_zero=False), _forgetting_events(acc, start_at_zero=False))))
        # learning from untrained: 1 event at start if all ones
        npt.assert_array_equal(_learning_events(self.ones, start_at_zero=True)[..., 0, :], 1)
        npt.assert_array_equal(_learning_events(self.ones, start_at_zero=True)[..., 1:, :], 0)
        # all other cases of all ones/zeros: no events
        npt.assert_array_equal(_learning_events(self.zeros, start_at_zero=False), 0)
        npt.assert_array_equal(_learning_events(self.zeros, start_at_zero=True), 0)
        npt.assert_array_equal(_learning_events(self.ones, start_at_zero=False), 0)
        # checkerboard case
        npt.assert_array_equal(_learning_events(self.checkerboard, dim=-1), self.checkerboard)

    def test_first_learn(self):
        self._common_tests(first_learn)
        # check edge cases
        npt.assert_array_equal(first_learn(self.ones), 0)
        npt.assert_array_equal(first_learn(self.zeros), self.n_steps)
        npt.assert_array_equal(first_learn(self.zero_to_ones), 1)
        npt.assert_array_equal(first_learn(self.one_to_zeros), 0)
        # checkerboard case
        npt.assert_array_equal(first_learn(self.checkerboard, dim=-1), [0, 1])

    def test_first_unforgettable(self):
        self._common_tests(first_unforgettable)
        # check edge cases
        npt.assert_array_equal(first_unforgettable(self.ones), 0)
        npt.assert_array_equal(first_unforgettable(self.zeros), self.n_steps)
        npt.assert_array_equal(first_unforgettable(self.zero_to_ones), 1)
        npt.assert_array_equal(first_unforgettable(self.one_to_zeros), self.n_steps)
        # checkerboard case
        npt.assert_array_equal(first_unforgettable(self.checkerboard, dim=-1), [6, 7])

    def test_first_forget(self):
        self._common_tests(first_forget)
        # check edge cases
        npt.assert_array_equal(first_forget(self.ones), self.n_steps)
        npt.assert_array_equal(first_forget(self.zeros), 0)
        npt.assert_array_equal(first_forget(self.zero_to_ones), 0)
        npt.assert_array_equal(first_forget(self.one_to_zeros), 1)
        # checkerboard case
        npt.assert_array_equal(first_forget(self.checkerboard, dim=-1), [1, 0])

    def test_first_unlearnable(self):
        self._common_tests(first_unlearnable)
        # check edge cases
        npt.assert_array_equal(first_unlearnable(self.ones), self.n_steps)
        npt.assert_array_equal(first_unlearnable(self.zeros), 0)
        npt.assert_array_equal(first_unlearnable(self.zero_to_ones), self.n_steps)
        npt.assert_array_equal(first_unlearnable(self.one_to_zeros), 1)
        # checkerboard case
        npt.assert_array_equal(first_unlearnable(self.checkerboard, dim=-1), [7, 6])

    def test_count_forgetting(self):
        self._common_tests(count_forgetting)
        for acc in self.acc:  # check that |learning - forgetting| \leq 1
            count_learning = torch.count_nonzero(_learning_events(acc), dim=-2)
            npt.assert_array_less(torch.abs(count_learning - count_forgetting(acc)), 2)
        # check edge cases
        npt.assert_array_equal(count_forgetting(self.ones), 0)
        npt.assert_array_equal(count_forgetting(self.zeros), 0)
        npt.assert_array_equal(count_forgetting(self.zero_to_ones), 0)
        npt.assert_array_equal(count_forgetting(self.one_to_zeros), 1)
        # checkerboard case
        npt.assert_array_equal(count_forgetting(self.checkerboard, dim=-1), [3, 3])

    def test_is_unforgettable(self):
        self._common_tests(is_unforgettable)
        # check edge cases
        npt.assert_array_equal(is_unforgettable(self.ones), 1)
        npt.assert_array_equal(is_unforgettable(self.zeros), 0)
        npt.assert_array_equal(is_unforgettable(self.zero_to_ones), 1)
        npt.assert_array_equal(is_unforgettable(self.one_to_zeros), 0)
        # checkerboard case
        npt.assert_array_equal(is_unforgettable(self.checkerboard, dim=-1), [0, 0])


    def _test_online_forgetting(self, Class, functional):
        # test without batches, assuming (T, ..., N)
        for acc in self.acc:
            obj = Class(acc.shape[-1])  # assumes last dim is batch dim
            for step in acc:
                obj.add(step)
            npt.assert_array_equal(obj.get(), functional(acc, dim=0))
        # test with batches, assuming (..., T, N)
        for acc, batch_size in zip(self.acc, range(1, len(self.acc))):
            obj = Class(acc.shape[-1])
            for i in range(acc.shape[-2]):
                step = acc[..., i, :]
                batches = torch.split(torch.randperm(step.shape[-1]), batch_size)
                for batch in batches:
                    obj.add(step[..., batch], minibatch_idx=batch)
            npt.assert_array_equal(obj.get(), functional(acc, dim=-2))


    def test_online_count_forgetting(self):
        self._test_online_forgetting(OnlineCountForgetting, count_forgetting)


if __name__ == '__main__':
    unittest.main()
