import torch
from typing import Union

"""
Author: https://gist.github.com/spezold/42a451682422beb42bc43ad0c0967a30
"""
def percentile(t: torch.Tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattenepip d input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values
