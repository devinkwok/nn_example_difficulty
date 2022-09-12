from typing import List
import unittest
import numpy as np
import numpy.testing as npt

from difficulty.metrics import *


class TestMetrics(unittest.TestCase):


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

    def test_ArgsUnchanged(self):
        logit = self.logits[0]
        idx = np.zeros(len(logit.shape), dtype=int)
        with self.ArgsUnchanged(logit):
            np.put(np.copy(logit), idx, 0.)
        try:  # modifying array in place should raise exception
            with self.ArgsUnchanged(logit):
                np.put(logit, idx, 0.)
            self.assertFalse()
        except:
            pass


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
            np.random.randn(I, N, C),
            np.random.randn(R, I, N, C),
            np.random.randn(R, I, S, N, C),
        ]
        self.labels = [
            np.random.randint(0, C, (I, N)),
            np.random.randint(0, C, (R, I, N)),
            np.random.randint(0, C, (R, I, S, N)),
        ]
        self.zero_labels = [
            np.zeros([I, N], dtype=int),
            np.zeros([R, I, N], dtype=int),
            np.zeros([R, I, S, N], dtype=int),
        ]
        self.acc = [zero_one_accuracy(x, y) for x, y in zip(self.logits, self.labels)]

        self.zeros = np.zeros([I, N], dtype=bool)
        self.ones = np.ones([I, N], dtype=bool)
        self.zero_to_ones = np.concatenate([self.zeros[..., 0:1, :], self.ones[..., 1:, :]], axis=-2)
        self.one_to_zeros = np.logical_not(self.zero_to_ones)
        self.never_learned = forgetting_events(self.zeros)
        self.always_learned = forgetting_events(self.ones)
        self.learn_after_start = forgetting_events(self.zero_to_ones)
        self.forget_after_start = forgetting_events(self.one_to_zeros)


    def test_softmax(self):
        for logit in self.logits:
            with self.ArgsUnchanged(logit):
                x = softmax(logit)
            self.assertEqual(x.shape, logit.shape)
            npt.assert_array_less(0, x)
            npt.assert_array_less(x, 1)
            npt.assert_array_equal(np.argmax(x, axis=-1), np.argmax(logit, axis=-1))
            npt.assert_allclose(np.sum(x, axis=-1), np.ones_like(logit[..., 0]))

    def test_zero_one_accuracy(self):
        for logit, label in zip(self.logits, self.labels):
            with self.ArgsUnchanged(logit, label):
                x = zero_one_accuracy(logit, label)
            self.assertEqual(x.shape, logit.shape[:-1])

    def test_entropy(self):
        for logit, label in zip(self.logits, self.labels):
            with self.ArgsUnchanged(logit, label):
                x = entropy(logit, label)
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x)

    def test_class_confidence(self):
        for logit, label, zeros in zip(self.logits, self.labels, self.zero_labels):
            prob = softmax(logit)
            with self.ArgsUnchanged(prob, label):
                x = class_confidence(prob, label)
            self.assertEqual(x.shape, logit.shape[:-1])
            x = class_confidence(prob, zeros)
            npt.assert_array_equal(x, prob[..., 0])

    def test_max_confidence(self):
        for logit in self.logits:
            prob = softmax(logit)
            with self.ArgsUnchanged(prob):
                x = max_confidence(prob)
            self.assertEqual(x.shape, logit.shape[:-1])
            self.assertFalse(np.any(np.broadcast_to(np.expand_dims(x, -1), logit.shape) < prob))

    def test_margin(self):
        for logit, label, acc in zip(self.logits, self.labels, self.acc):
            prob = softmax(logit)
            with self.ArgsUnchanged(prob, label):
                x = margin(prob, label)
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x[acc])
            npt.assert_array_less(x[np.logical_not(acc)], 0)

    def test_error_l2_norm(self):
        for logit, label in zip(self.logits, self.labels):
            prob = softmax(logit)
            with self.ArgsUnchanged(prob, label):
                x = error_l2_norm(prob, label)
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x)
            # max error is prob 1 on wrong class
            # hence, error = \sqrt{1^2 + 1^2 + 0 + \dots}
            npt.assert_array_less(x, np.sqrt(2))
        all_correct = error_l2_norm(self.one_to_zeros.T, np.zeros(self.n_examples, dtype=int))
        npt.assert_array_equal(all_correct, 0)
        all_wrong = error_l2_norm(self.one_to_zeros.T, np.ones(self.n_examples, dtype=int))
        npt.assert_array_equal(all_wrong, np.sqrt(2))

    def test_forgetting(self):
        for acc in self.acc:
            with self.ArgsUnchanged(acc):
                x = forgetting_events(acc)
            target_shape = list(acc.shape)
            target_shape[-2] += 1
            self.assertEqual(x.shape, tuple(target_shape))
        # check edge cases
        npt.assert_array_equal(self.always_learned[..., 0, :], 1)
        npt.assert_array_equal(self.always_learned[..., 1:, :], 0)
        npt.assert_array_equal(self.never_learned[..., -1, :], 1)
        npt.assert_array_equal(self.never_learned[..., :-1, :], 0)

    def test_count_forgetting(self):
        for acc in self.acc:
            x = forgetting_events(acc)
            with self.ArgsUnchanged(x):
                a = count_events_over_steps(x, 1)
                b = count_events_over_steps(x, 0)
                c = count_events_over_steps(x, -1)
            self.assertEqual(a.shape, x.shape[:-2] + x.shape[-1:])
            npt.assert_array_equal(a + b + c, acc.shape[-2] + 1)
            npt.assert_array_equal(count_forgetting(x), c)
        # check edge cases
        npt.assert_array_equal(count_forgetting(self.always_learned), 0)
        npt.assert_array_equal(count_forgetting(self.never_learned), 0)
        npt.assert_array_equal(count_forgetting(self.learn_after_start), 0)
        npt.assert_array_equal(count_forgetting(self.forget_after_start), 1)

    def test_first_learn(self):
        for acc in self.acc:
            x = forgetting_events(acc)
            with self.ArgsUnchanged(x):
                first = first_learn(x)
            self.assertEqual(first.shape, x.shape[:-2] + x.shape[-1:])
            npt.assert_array_less(first, acc.shape[-2] + 1)
        # check edge cases
        npt.assert_array_equal(first_learn(self.always_learned), 0)
        npt.assert_array_equal(first_learn(self.never_learned), self.n_steps)
        npt.assert_array_equal(first_learn(self.learn_after_start), 1)
        npt.assert_array_equal(first_learn(self.forget_after_start), 0)

    def test_is_unforgettable(self):
        for acc in self.acc:
            x = forgetting_events(acc)
            with self.ArgsUnchanged(x):
                forget = is_unforgettable(x)
            target_shape = x.shape[:-2] + x.shape[-1:]
            self.assertEqual(forget.shape, target_shape)
        # check edge cases
        npt.assert_array_equal(is_unforgettable(self.always_learned), 1)
        npt.assert_array_equal(is_unforgettable(self.never_learned), 0)
        npt.assert_array_equal(is_unforgettable(self.learn_after_start), 1)
        npt.assert_array_equal(is_unforgettable(self.forget_after_start), 0)

    def test_first_unforgettable(self):
        for acc in self.acc:
            x = forgetting_events(acc)
            with self.ArgsUnchanged(x):
                first = first_unforgettable(x)
            self.assertEqual(first.shape, x.shape[:-2] + x.shape[-1:])
        # check edge cases
        npt.assert_array_equal(first_unforgettable(self.always_learned), 0)
        npt.assert_array_equal(first_unforgettable(self.never_learned), self.n_steps)
        npt.assert_array_equal(first_unforgettable(self.learn_after_start), 1)
        npt.assert_array_equal(first_unforgettable(self.forget_after_start), self.n_steps)

    def test_perturb_forgetting_events(self):
        for acc in self.acc:
            with self.ArgsUnchanged(acc):
                x = perturb_forgetting_events(acc)
            target_shape = list(acc.shape)
            target_shape[-2] += 1
            self.assertEqual(x.shape, tuple(target_shape))
        # check edge cases
        npt.assert_array_equal(perturb_forgetting_events(self.ones)[..., :-1, :], 0)
        npt.assert_array_equal(perturb_forgetting_events(self.ones)[..., -1, :], -1)
        npt.assert_array_equal(perturb_forgetting_events(self.zeros)[..., 0, :], -1)
        npt.assert_array_equal(perturb_forgetting_events(self.zeros)[..., 1:, :], 0)

    def test_perturb_first_forget(self):
        for acc in self.acc:
            x = perturb_forgetting_events(acc)
            with self.ArgsUnchanged(x):
                first = perturb_first_forget(x)
            self.assertEqual(first.shape, x.shape[:-2] + x.shape[-1:])
        # check edge cases
        npt.assert_array_equal(perturb_first_forget(perturb_forgetting_events(self.ones)), self.n_steps)
        npt.assert_array_equal(perturb_first_forget(perturb_forgetting_events(self.zeros)), 0)
        npt.assert_array_equal(perturb_first_forget(perturb_forgetting_events(self.zero_to_ones)), 0)
        npt.assert_array_equal(perturb_first_forget(perturb_forgetting_events(self.one_to_zeros)), 1)

    def test_rank(self):
        for logit in self.logits:
            with self.ArgsUnchanged(logit):
                x = rank(logit)
            npt.assert_array_equal(np.argsort(x, axis=-1), np.argsort(logit, axis=-1))


if __name__ == '__main__':
    unittest.main()
