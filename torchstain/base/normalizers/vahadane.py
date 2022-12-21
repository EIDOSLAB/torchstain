def VahadaneNormalizer(backend='numpy'):
    if backend == 'numpy':
        from torchstain.numpy.normalizers import VahadaneReinhardNormalizer
        return NumpyVahadaneNormalizer()
    elif backend == "torch":
        raise NotImplementedError
    elif backend == "tensorflow":
        raise NotImplementedError
    else:
        raise Exception(f'Unknown backend {backend}')
