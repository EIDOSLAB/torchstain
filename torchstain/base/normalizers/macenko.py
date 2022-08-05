def MacenkoNormalizer(backend='torch'):
    if backend == 'numpy':
        from torchstain.numpy.normalizers import NumpyMacenkoNormalizer
        return NumpyMacenkoNormalizer()
    elif backend == "torch":
        from torchstain.torch_backend.normalizers.macenko import TorchMacenkoNormalizer
        return TorchMacenkoNormalizer()
    elif backend == "tensorflow":
        from torchstain.tf_backend.normalizers.macenko import TensorFlowMacenkoNormalizer
        return TensorFlowMacenkoNormalizer()
    else:
        raise Exception(f'Unknown backend {backend}')
