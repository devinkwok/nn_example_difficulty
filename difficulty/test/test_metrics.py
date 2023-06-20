from typing import List
import unittest
import numpy as np
import numpy.testing as npt

from difficulty.metrics import *


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


class TestMetrics(unittest.TestCase):

    def test_ArgsUnchanged(self):
        logit = self.logits[0]
        idx = np.zeros(len(logit.shape), dtype=int)
        with ArgsUnchanged(logit):
            np.put(np.copy(logit), idx, 0.)
        try:  # modifying array in place should raise exception
            with ArgsUnchanged(logit):
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
        self.zero_to_ones = torch.concatenate([self.zeros[..., 0:1, :], self.ones[..., 1:, :]], axis=-2)
        self.one_to_zeros = 1 - self.zero_to_ones


    def test_softmax(self):
        for logit in self.logits:
            with ArgsUnchanged(logit):
                x = softmax(logit).numpy()
            self.assertEqual(x.shape, logit.shape)
            npt.assert_array_less(0, x)
            npt.assert_array_less(x, 1)
            npt.assert_array_equal(np.argmax(x, axis=-1), np.argmax(logit, axis=-1))
            npt.assert_allclose(np.sum(x, axis=-1), np.ones_like(logit[..., 0]), rtol=1e-6)

    def test_zero_one_accuracy(self):
        for logit, label in zip(self.logits, self.labels):
            with ArgsUnchanged(logit, label):
                x = zero_one_accuracy(logit, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])

    def test_entropy(self):
        for logit, label in zip(self.logits, self.labels):
            with ArgsUnchanged(logit, label):
                x = entropy(logit, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x)

    def test_class_confidence(self):
        for logit, label, zeros in zip(self.logits, self.labels, self.zero_labels):
            prob = softmax(logit)
            with ArgsUnchanged(prob, label):
                x = class_confidence(prob, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            x = class_confidence(prob, zeros)
            npt.assert_array_equal(x, prob[..., 0])

    def test_max_confidence(self):
        for logit in self.logits:
            prob = softmax(logit)
            with ArgsUnchanged(prob):
                x = max_confidence(prob).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            self.assertFalse(np.any(np.broadcast_to(np.expand_dims(x, -1), logit.shape) < prob.numpy()))

    def test_margin(self):
        for logit, label, acc in zip(self.logits, self.labels, self.acc):
            prob = softmax(logit)
            with ArgsUnchanged(prob, label):
                x = margin(prob, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x[acc])
            npt.assert_array_less(x[np.logical_not(acc.numpy())], 0)

    def test_error_l2_norm(self):
        for logit, label in zip(self.logits, self.labels):
            prob = softmax(logit)
            with ArgsUnchanged(prob, label):
                x = error_l2_norm(prob, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x)
            # max error is prob 1 on wrong class
            # hence, error = \sqrt{1^2 + 1^2 + 0 + \dots}
            npt.assert_array_less(x, np.sqrt(2))
        all_correct = error_l2_norm(self.one_to_zeros.T, torch.zeros(self.n_examples, dtype=int)).numpy()
        npt.assert_array_equal(all_correct, 0)
        all_wrong = error_l2_norm(self.one_to_zeros.T, torch.ones(self.n_examples, dtype=int)).numpy()
        npt.assert_array_equal(all_wrong, np.sqrt(2))

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

    def test_rank(self):
        for logit in self.logits:
            with ArgsUnchanged(logit):
                x = rank(logit).numpy()
            npt.assert_array_equal(np.argsort(x, axis=-1), np.argsort(logit, axis=-1))


if __name__ == '__main__':
    unittest.main()
