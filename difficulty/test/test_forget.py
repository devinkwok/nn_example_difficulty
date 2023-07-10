import unittest
import numpy.testing as npt

from difficulty.metrics import *
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

    def test_forgetting(self):
        for acc in self.acc:
            with ArgsUnchanged(acc):
                x = forgetting_events(acc).numpy()
            target_shape = list(acc.shape)
            target_shape[-2] += 1
            self.assertEqual(x.shape, tuple(target_shape))
        # check edge cases
        npt.assert_array_equal(learning_events(self.ones)[..., 0, :].numpy(), 1)
        npt.assert_array_equal(forgetting_events(self.ones)[..., 0, :].numpy(), 0)
        npt.assert_array_equal(learning_events(self.ones)[..., 1:, :].numpy(), 0)
        npt.assert_array_equal(forgetting_events(self.ones)[..., 1:, :].numpy(), 0)
        npt.assert_array_equal(learning_events(self.zeros)[..., -1, :].numpy(), 1)
        npt.assert_array_equal(forgetting_events(self.zeros)[..., -1, :].numpy(), 0)
        npt.assert_array_equal(learning_events(self.zeros)[..., :-1, :].numpy(), 0)
        npt.assert_array_equal(forgetting_events(self.zeros)[..., :-1, :].numpy(), 0)

    def test_count_forgetting(self):
        for acc in self.acc:
            forget_events = forgetting_events(acc)
            learn_events = learning_events(acc)
            a = count_events(learn_events)
            b = count_events(torch.logical_and(torch.logical_not(learn_events), torch.logical_not(forget_events)))
            c = count_events(forget_events)
            self.assertEqual(a.shape, forget_events.shape[:-2] + forget_events.shape[-1:])
            npt.assert_array_equal(a + b + c, acc.shape[-2] + 1)
            npt.assert_array_equal(count_forgetting(acc), c)
        # check edge cases
        npt.assert_array_equal(count_forgetting(self.ones), 0)
        npt.assert_array_equal(count_forgetting(self.zeros), 0)
        npt.assert_array_equal(count_forgetting(self.zero_to_ones), 0)
        npt.assert_array_equal(count_forgetting(self.one_to_zeros), 1)

    def test_first_learn(self):
        for acc in self.acc:
            with ArgsUnchanged(acc):
                first = first_learn(acc)
            self.assertEqual(first.shape, acc.shape[:-2] + acc.shape[-1:])
            npt.assert_array_less(first, acc.shape[-2] + 1)
        # check edge cases
        npt.assert_array_equal(first_learn(self.ones), 0)
        npt.assert_array_equal(first_learn(self.zeros), self.n_steps)
        npt.assert_array_equal(first_learn(self.zero_to_ones), 1)
        npt.assert_array_equal(first_learn(self.one_to_zeros), 0)

    def test_is_unforgettable(self):
        for acc in self.acc:
            x = learning_events(acc)
            y = forgetting_events(acc)
            with ArgsUnchanged(x):
                with ArgsUnchanged(y):
                    forget = is_unforgettable(x, y)
            target_shape = x.shape[:-2] + x.shape[-1:]
            self.assertEqual(forget.shape, target_shape)
        # check edge cases
        npt.assert_array_equal(is_unforgettable(
            learning_events(self.ones), forgetting_events(self.ones)), 1)
        npt.assert_array_equal(is_unforgettable(
            learning_events(self.zeros), forgetting_events(self.zeros)), 0)
        npt.assert_array_equal(is_unforgettable(
            learning_events(self.zero_to_ones), forgetting_events(self.zero_to_ones)), 1)
        npt.assert_array_equal(is_unforgettable(
            learning_events(self.one_to_zeros), forgetting_events(self.one_to_zeros)), 0)

    def test_first_unforgettable(self):
        for acc in self.acc:
            with ArgsUnchanged(acc):
                first = first_unforgettable(acc)
            self.assertEqual(first.shape, acc.shape[:-2] + acc.shape[-1:])
        # check edge cases
        npt.assert_array_equal(first_unforgettable(self.ones), 0)
        npt.assert_array_equal(first_unforgettable(self.zeros), self.n_steps)
        npt.assert_array_equal(first_unforgettable(self.zero_to_ones), 1)
        npt.assert_array_equal(first_unforgettable(self.one_to_zeros), self.n_steps)

    def test_perturb_forgetting_events(self):
        for acc in self.acc:
            with ArgsUnchanged(acc):
                x = perturb_forgetting_events(acc).numpy()
            target_shape = list(acc.shape)
            target_shape[-2] += 1
            self.assertEqual(x.shape, tuple(target_shape))
        # check edge cases
        npt.assert_array_equal(perturb_learning_events(self.ones)[..., :-1, :], 0)
        npt.assert_array_equal(perturb_learning_events(self.ones)[..., -1, :], 0)
        npt.assert_array_equal(perturb_forgetting_events(self.ones)[..., :-1, :], 0)
        npt.assert_array_equal(perturb_forgetting_events(self.ones)[..., -1, :], 1)
        npt.assert_array_equal(perturb_learning_events(self.zeros)[..., 0, :], 0)
        npt.assert_array_equal(perturb_learning_events(self.zeros)[..., 1:, :], 0)
        npt.assert_array_equal(perturb_forgetting_events(self.zeros)[..., 0, :], 1)
        npt.assert_array_equal(perturb_forgetting_events(self.zeros)[..., 1:, :], 0)

    def test_perturb_first_forget(self):
        for acc in self.acc:
            with ArgsUnchanged(acc):
                first = perturb_first_forget(acc)
            self.assertEqual(first.shape, acc.shape[:-2] + acc.shape[-1:])
        # check edge cases
        npt.assert_array_equal(perturb_first_forget(self.ones), self.n_steps)
        npt.assert_array_equal(perturb_first_forget(self.zeros), 0)
        npt.assert_array_equal(perturb_first_forget(self.zero_to_ones), 0)
        npt.assert_array_equal(perturb_first_forget(self.one_to_zeros), 1)
