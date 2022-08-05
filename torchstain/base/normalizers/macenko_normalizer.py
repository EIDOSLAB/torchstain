from .numpy_macenko_normalizer import NumpyMacenkoNormalizer


def MacenkoNormalizer(backend='torch'):
    if backend == 'numpy':
        return NumpyMacenkoNormalizer()
    elif backend == "torch":
        from torchstain.torch_backend.normalizers.torch_macenko_normalizer import TorchMacenkoNormalizer
        return TorchMacenkoNormalizer()
    elif backend == "tensorflow":
        from torchstain.tf_backend.normalizers.tensorflow_macenko_normalizer import TensorFlowMacenkoNormalizer
        return TensorFlowMacenkoNormalizer()
    else:
        raise Exception(f'Unknown backend {backend}')
