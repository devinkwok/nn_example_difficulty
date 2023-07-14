import unittest
from itertools import product
import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn

from difficulty.test.base import BaseTest
from difficulty.metrics import *


class TestModel(BaseTest):

    def setUp(self):
        super().setUp()
        self.linear = self._make_linear_models()

    def _make_linear_models(self, n_models=1, weights=None, scale=1.):
        models = []
        for i in range(n_models):
            linear = nn.Linear(self.n_inputs, self.n_outputs)
            if weights is None:
                weights = linear.weight.data
            else:
                linear.weight.data = weights.detach().clone() * scale**i
            # non-zero bias makes bigger gradients
            linear.bias.data = torch.full_like(linear.bias.data, 1)
            models.append(nn.Sequential(nn.Flatten(), linear))
        if n_models < 2:
            return models[0]
        return models

    def test_input_gradient(self):
        grad = input_gradient(self.model, self.data, self.data_labels)
        self.assertEqual(grad.shape, self.data.shape)
        # gradients should be identical regardless of batch size
        grad_batch = torch.cat([input_gradient(self.model, x.unsqueeze(0), y.unsqueeze(0)) for x, y in zip(self.data, self.data_labels)], dim=0)
        self.all_close(grad, grad_batch)
        grad_batch = input_gradient_from_dataloader(self.model, self.dataloader, device="cpu")
        self.all_close(grad, grad_batch)
        # linear model has linear response to input, so long as softmax is excluded from loss
        grad = input_gradient(self.linear, self.data, torch.zeros_like(self.data_labels), loss_fn=ClassOutput(softmax=False))
        step_size = 1.
        greater = self.data + step_size * grad
        npt.assert_array_less(self.linear(self.data)[..., 0].detach(), self.linear(greater)[..., 0].detach())
        # check that argmax labels are being applied
        grad = input_gradient(self.linear, self.data, loss_fn=ClassOutput(softmax=False, use_argmax_labels=True))
        step_size = 1.
        greater = self.data + step_size * grad
        npt.assert_array_less(np.max(self.linear(self.data).detach().numpy(), axis=-1),
                              np.max(self.linear(greater).detach().numpy(), axis=-1))

    def test_variance_of_gradients(self):
        # variance should be zero for same model
        vog = variance_of_gradients([self.model]*5, self.dataloader)
        self.assertEqual(vog.shape, (self.n,))
        self.all_close(vog, torch.zeros_like(vog))
        # variance should not depend on dataloader
        models = self._make_linear_models(6, scale=-1.5)
        vog = torch.cat([variance_of_gradients(models, self.dataloader)])
        vog_1 = variance_of_gradients(models, [(self.data, self.data_labels)])
        self.all_close(vog, vog_1)
        vog_2 = torch.cat([variance_of_gradients(models, [(x.unsqueeze(0), y.unsqueeze(0))]) for x, y in zip(self.data, self.data_labels)])
        self.all_close(vog, vog_2)
        # variance should not depend on color
        channel_weights = torch.randn(self.n_outputs, self.n_inputs // 3)
        channel_weights = torch.cat([channel_weights]*3, dim=1)
        ch_models = self._make_linear_models(6, weights=channel_weights, scale=-1.5)
        one_channel = self.data[:, 0, ...]
        zeros = torch.zeros_like(one_channel)
        r = torch.stack([one_channel, zeros, zeros], dim=1)
        gb = torch.stack([zeros, one_channel / 2, one_channel / 2], dim=1)
        vog_r = variance_of_gradients(ch_models, [(r, self.data_labels)])
        vog_gb = variance_of_gradients(ch_models, [(gb, self.data_labels)])
        self.all_close(mean_color_channels(r), mean_color_channels(gb))
        self.all_close(vog_r, vog_gb)
        npt.assert_array_less(-1e-8, vog_r)
        # variance should be per-pixel and not depend on per-pixel offset
        n_pixels = self.data.shape[-1] * self.data.shape[-2]
        pixel_weights = []
        for i in range(3):
            pixel_weights.append(torch.cat([torch.randn(
                self.n_outputs, self.n_inputs // 3 // n_pixels)]*n_pixels, dim=1))
        pixel_weights = torch.cat(pixel_weights, dim=1)
        px_models = self._make_linear_models(6, weights=pixel_weights, scale=-1.5)
        one_pixel = self.data[:, :, 0, 0]
        zeros = torch.zeros_like(one_pixel)
        x_1 = torch.stack([one_pixel] + [zeros] * (n_pixels - 1), dim=-1).reshape(*self.data.shape)
        x_2 = torch.stack([one_pixel] * n_pixels, dim=-1).reshape(*self.data.shape) / n_pixels
        self.all_close(px_models[0](x_1), px_models[0](x_2))
        vog_1 = variance_of_gradients(px_models, [(x_1, self.data_labels)])
        vog_2 = variance_of_gradients(px_models, [(x_2, self.data_labels)])
        self.all_close(vog_1, vog_2)
        npt.assert_array_less(-1e-8, vog_1)

    def _grad_flags_unaffected(self, apply_fn):
        for weight_grad, bias_grad in product([False, True], [False, True]):
            model = self._make_linear_models()
            # use optimizer to set requires_grad_(True)
            torch.optim.SGD(model.parameters(), lr=1)
            model[1].weight.requires_grad_(weight_grad)
            model[1].bias.requires_grad_(bias_grad)
            apply_fn(model)
            self.assertEqual(model[1].weight.requires_grad, weight_grad)
            self.assertEqual(model[1].bias.requires_grad, bias_grad)

    def _returns_output(self, apply_fn):

        def outputs_match(model, data):
            model.train()  # check that model.eval() is called
            _, out_1 = apply_fn(model, data)
            with torch.no_grad():
                model.eval()
                out = model(data)
            self.all_close(out, out_1)

        outputs_match(self.model, self.data)
        outputs_match(self.model, torch.ones_like(self.data))
        outputs_match(self._make_linear_models(), torch.ones_like(self.data))

    def test_online_vog(self):
        models = self._make_linear_models(6, scale=-1.5)
        vog = variance_of_gradients(models, self.dataloader)
        online_vog = OnlineVarianceOfGradients()
        for model in models:
            online_vog = online_vog.add(model, self.dataloader)
            online_vog.save(self.tmp_file)
            online_vog = OnlineVarianceOfGradients.load(self.tmp_file)
        self.all_close(vog, online_vog.get())
        # check that requires_grad flags are not affected
        self._grad_flags_unaffected(
            lambda m: variance_of_gradients([m, m], [(self.data, self.data_labels)]))
        # add() returns outputs that are equivalent to model(x)
        online_vog = OnlineVarianceOfGradients()
        self._returns_output(lambda m, d: online_vog.add(
            m, [(d, torch.zeros(len(d), dtype=torch.long))], return_output=True))

    def test_gradient_norm(self):
        # loss of 0 should give gradient norm of 0
        self.model.eval()
        est_labels = self.model(self.data).detach()
        grand = gradient_norm(self.model, self.data, est_labels, loss_fn=nn.MSELoss())
        self.assertTrue(torch.all(torch.less(torch.abs(grand), 1e-3)))
        # check that functional gradient is same as computing each data point's gradient individually
        grand = gradient_norm(self.model, self.data, self.data_labels)
        grand_1 = functional_gradient_norm(self.model, self.data, self.data_labels)
        self.all_close(grand, grand_1)
        grand_2 = grand_score(self.model, self.dataloader)
        self.all_close(grand, grand_2)
        # check that requires_grad flags are not affected
        self._grad_flags_unaffected(
            lambda m: grand_score(m, [(self.data, self.data_labels)]))
        # returns outputs that are equivalent to model(x)
        self._returns_output(lambda m, d: grand_score(
            m, [(d, torch.zeros(len(d), dtype=torch.long))], return_output=True))


if __name__ == '__main__':
    unittest.main()
