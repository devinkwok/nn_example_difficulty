import unittest
import numpy as np
import numpy.testing as npt

from difficulty.metrics import *
from difficulty.test.base import BaseTest


class TestMetrics(BaseTest):

    def test_softmax(self):
        for logit in self.logits:
            with self.ArgsUnchanged(logit):
                x = softmax(logit).numpy()
            self.assertEqual(x.shape, logit.shape)
            npt.assert_array_less(0, x)
            npt.assert_array_less(x, 1)
            npt.assert_array_equal(np.argmax(x, axis=-1), np.argmax(logit, axis=-1))
            self.all_close(np.sum(x, axis=-1), np.ones_like(logit[..., 0]))

    def test_zero_one_accuracy(self):
        for logit, label in zip(self.logits, self.logit_labels):
            with self.ArgsUnchanged(logit, label):
                x = zero_one_accuracy(logit, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])

    def test_entropy(self):
        for logit in self.logits:
            prob = softmax(logit)
            with self.ArgsUnchanged(prob):
                x = entropy(prob).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x)
            # entropy of deterministic variable is 0
            deterministic = torch.zeros(logit.shape[-1])
            deterministic[0] = 1.
            det_ent = entropy(deterministic)
            npt.assert_array_equal(0, det_ent)
            # entropy of uniform distribution is maximal
            uniform = entropy(torch.ones(logit.shape[-1]) / logit.shape[-1])
            npt.assert_array_less(x, uniform)

    def test_class_confidence(self):
        for logit, label, zeros in zip(self.logits, self.logit_labels, self.zero_labels):
            prob = softmax(logit)
            with self.ArgsUnchanged(prob, label):
                x = class_confidence(prob, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            x = class_confidence(prob, zeros)
            npt.assert_array_equal(x, prob[..., 0])

    def test_max_confidence(self):
        for logit in self.logits:
            prob = softmax(logit)
            with self.ArgsUnchanged(prob):
                x = max_confidence(prob).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            self.assertFalse(np.any(np.broadcast_to(np.expand_dims(x, -1), logit.shape) < prob.numpy()))

    def test_margin(self):
        for logit, label, acc in zip(self.logits, self.logit_labels, self.acc):
            prob = softmax(logit)
            with self.ArgsUnchanged(prob, label):
                x = margin(prob, label).numpy()
            self.assertEqual(x.shape, logit.shape[:-1])
            npt.assert_array_less(0, x[acc])
            npt.assert_array_less(x[np.logical_not(acc.numpy())], 0)

    def test_error_l2_norm(self):
        for logit, label in zip(self.logits, self.logit_labels):
            prob = softmax(logit)
            with self.ArgsUnchanged(prob, label):
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
            with self.ArgsUnchanged(logit):
                x = rank(logit).numpy()
            npt.assert_array_equal(np.argsort(x, axis=-1), np.argsort(logit, axis=-1))

    def test_convenience_function(self):
        for logit, label in zip(self.logits, self.logit_labels):
            results = pointwise_metrics(logit, label)
            prob = softmax(logit)
            self.all_close(results["acc"], zero_one_accuracy(logit, label))
            self.all_close(results["ent"], entropy(prob))
            self.all_close(results["conf"], class_confidence(prob, label))
            self.all_close(results["maxconf"], max_confidence(prob))
            self.all_close(results["margin"], margin(prob, label))
            self.all_close(results["el2n"], error_l2_norm(prob, label))
            # check that the following run without import issues
            create_online_pointwise_metrics()


if __name__ == '__main__':
    unittest.main()
