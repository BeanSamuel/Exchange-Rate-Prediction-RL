import numpy as np
import torch
from torch import nn

from utils.torch_utils import to_torch_size


def eval_no_grad(func):
    def _eval_no_grad(self, *args, **kwargs):
        if not self.training:
            with torch.no_grad():
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return _eval_no_grad


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        return x


def neg_log_p(x, mean, log_std):
    return 0.5 * (((x - mean) / torch.exp(log_std)) ** 2).sum(dim=-1) \
        + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
        + log_std.sum(dim=-1)


class RunningMeanStd(nn.Module):
    def __init__(self, in_size, eps=1e-05):
        super().__init__()
        self.in_size = to_torch_size(in_size)
        self.eps = eps

        self.register_buffer("mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)

        self.count += batch_count
        self.mean[:] = self.mean + delta * batch_count / self.count
        self.var[:] = m2 / self.count

    def forward(self, x, unnorm=False):
        if x.nelement() == 0:
            return x

        if self.training and not unnorm:
            axis = list(range(x.ndim - len(self.in_size)))
            mean = x.mean(axis)
            var = x.var(axis, correction=0)
            count = x.shape[:-1].numel()
            self._update(mean, var, count)

        if unnorm:
            y = torch.clamp(x, min=-5.0, max=5.0)
            y = torch.sqrt(self.var.float() + self.eps) * y + self.mean.float()
        else:
            y = (x - self.mean.float()) / torch.sqrt(self.var.float() + self.eps)
            y = torch.clamp(y, min=-5.0, max=5.0)

        return y
