from torchstain.normalizers.numpy_macenko_normalizer import NumpyMacenkoNormalizer
from torchstain.normalizers.torch_macenko_normalizer import TorchMacenkoNormalizer

def MacenkoNormalizer(backend='torch'):
    if backend not in ['torch', 'numpy']:
        raise Exception(f'Unkown backend {backend}')

    if backend == 'numpy':
        return NumpyMacenkoNormalizer()
    return TorchMacenkoNormalizer()
