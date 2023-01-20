import torch

def csplit(I):
    return [I[i] for i in range(I.shape[0])]

def cmerge(I1, I2, I3):
    return torch.stack([I1, I2, I3], dim=0)

def lab_split(I):
    I = I.type(torch.float32)
    I1, I2, I3 = csplit(I)
    return I1 / 2.55, I2 - 128, I3 - 128

def lab_merge(I1, I2, I3):
    return cmerge(I1 * 2.55, I2 + 128, I3 + 128)
