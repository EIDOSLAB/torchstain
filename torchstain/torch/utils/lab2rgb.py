import torch
from torchstain.torch.utils.rgb2lab import _rgb2xyz, _white

_xyz2rgb = torch.linalg.inv(_rgb2xyz)

def lab2rgb(lab):
    lab = lab.type(torch.float32)
    
    # rescale back from OpenCV format and extract LAB channel
    L, a, b = lab[0] / 2.55, lab[1] - 128, lab[2] - 128

    # vector scaling to produce X, Y, Z
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    # merge back to get reconstructed XYZ color image
    out = torch.stack([x, y, z], axis=0)

    # apply boolean transforms
    mask = out > 0.2068966
    not_mask = torch.logical_not(mask)
    out.masked_scatter_(mask, torch.pow(torch.masked_select(out, mask), 3))
    out.masked_scatter_(not_mask, (torch.masked_select(out, not_mask) - 16 / 116) / 7.787)

    # rescale to the reference white (illuminant)
    out = torch.mul(out, _white.type(out.dtype).unsqueeze(dim=-1).unsqueeze(dim=-1))

    # convert XYZ -> RGB color domain
    arr = torch.tensordot(out, torch.t(_xyz2rgb).type(out.dtype), dims=([0], [0]))
    mask = arr > 0.0031308
    not_mask = torch.logical_not(mask)
    arr.masked_scatter_(mask, 1.055 * torch.pow(torch.masked_select(arr, mask), 1 / 2.4) - 0.055)
    arr.masked_scatter_(not_mask, torch.masked_select(arr, not_mask) * 12.92)
    return torch.clamp(arr, 0, 1)
