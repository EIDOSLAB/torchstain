import torch

# constant conversion matrices between color spaces: https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
_rgb2xyz = torch.tensor([[0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]])
                        
_white = torch.tensor([0.95047, 1., 1.08883])

def rgb2lab(rgb):
    arr = rgb.type(torch.float32)

    # convert rgb -> xyz color domain
    mask = arr > 0.04045
    not_mask = torch.logical_not(mask)
    arr.masked_scatter_(mask, torch.pow((torch.masked_select(arr, mask) + 0.055) / 1.055, 2.4))
    arr.masked_scatter_(not_mask, torch.masked_select(arr, not_mask) / 12.92)

    xyz = torch.tensordot(torch.t(_rgb2xyz), arr, dims=([0], [0]))

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = torch.mul(xyz, 1 / _white.type(xyz.dtype).unsqueeze(dim=-1).unsqueeze(dim=-1))

    # nonlinear distortion and linear transformation
    mask = arr > 0.008856
    not_mask = torch.logical_not(mask)
    arr.masked_scatter_(mask, torch.pow(torch.masked_select(arr, mask), 1 / 3))
    arr.masked_scatter_(not_mask, 7.787 * torch.masked_select(arr, not_mask) + 16 / 166)

    # get each channel as individual tensors
    x, y, z = arr[0], arr[1], arr[2]

    # vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    # OpenCV format
    L *= 2.55
    a += 128
    b += 128

    # finally, get LAB color domain
    return torch.stack([L, a, b], axis=0)
