import torch
import torchstain
import numpy as np

def test_cov():
    x = np.random.randn(10, 10)
    cov_np = np.cov(x)
    cov_t = torchstain.torch_backend.utils.cov(torch.tensor(x))

    np.testing.assert_almost_equal(cov_np, cov_t.numpy())

def test_percentile():
    x = np.random.randn(10, 10)
    p = 20
    p_np = np.percentile(x, p, interpolation='nearest')
    p_t = torchstain.torch_backend.utils.percentile(torch.tensor(x), p)

    np.testing.assert_almost_equal(p_np, p_t)
