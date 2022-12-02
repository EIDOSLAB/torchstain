def ReinhardNormalizer(backend='numpy'):
    if backend == 'numpy':
        from torchstain.numpy.normalizers import NumpyReinhardNormalizer
        return NumpyReinhardNormalizer()
    elif backend == "torch":
        raise NotImplementedError
    elif backend == "tensorflow":
        from torchstain.tf.normalizers import TensorFlowReinhardNormalizer
        return TensorFlowReinhardNormalizer()
    else:
        raise Exception(f'Unknown backend {backend}')
