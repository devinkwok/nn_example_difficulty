import unittest
import numpy as np
import numpy.testing as npt

from difficulty.metrics import *
from difficulty.test.utils import ArgsUnchanged


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
        self.zero_to_ones = torch.cat([self.zeros[..., 0:1, :], self.ones[..., 1:, :]], axis=-2)
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

    def test_rank(self):
        for logit in self.logits:
            with ArgsUnchanged(logit):
                x = rank(logit).numpy()
            npt.assert_array_equal(np.argsort(x, axis=-1), np.argsort(logit, axis=-1))


if __name__ == '__main__':
    unittest.main()
