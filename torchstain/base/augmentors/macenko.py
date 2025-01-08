def MacenkoAugmentor(backend='torch', sigma1=0.2, sigma2=0.2):
    if backend == 'numpy':
        from torchstain.numpy.augmentors import NumpyMacenkoAugmentor
        return NumpyMacenkoAugmentor(sigma1=sigma1, sigma2=sigma2)
    elif backend == "torch":
        from torchstain.torch.augmentors import TorchMacenkoAugmentor
        return TorchMacenkoAugmentor(sigma1=sigma1, sigma2=sigma2)
    elif backend == "tensorflow":
        from torchstain.tf.augmentors import TensorFlowMacenkoAugmentor
        return TensorFlowMacenkoAugmentor(sigma1=sigma1, sigma2=sigma2)
    else:
        raise Exception(f'Unknown backend {backend}')
