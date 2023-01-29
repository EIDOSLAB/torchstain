def MacenkoNormalizer(backend='torch'):
    if backend == 'numpy':
        from torchstain.numpy.normalizers import NumpyMacenkoNormalizer
        return NumpyMacenkoNormalizer()
    elif backend == "torch":
        from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer
        return TorchMacenkoNormalizer()
    elif backend == "tensorflow":
        from torchstain.tf.normalizers.macenko import TensorFlowMacenkoNormalizer
        return TensorFlowMacenkoNormalizer()
    elif backend == "jax":
        from torchstain.jax.normalizers.macenko import JaxMacenkoNormalizer
        return JaxMacenkoNormalizer()
    else:
        raise Exception(f'Unknown backend {backend}')
