def ReinhardNormalizer(backend='numpy', method=None):
    if backend == 'numpy':
        from torchstain.numpy.normalizers import NumpyReinhardNormalizer
        return NumpyReinhardNormalizer(method=method)
    elif backend == "torch":
        from torchstain.torch.normalizers import TorchReinhardNormalizer
        return TorchReinhardNormalizer(method=method)
    elif backend == "tensorflow":
        from torchstain.tf.normalizers import TensorFlowReinhardNormalizer
        return TensorFlowReinhardNormalizer(method=method)
    else:
        raise Exception(f'Unknown backend {backend}')
