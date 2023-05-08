def MacenkoAugmentor(backend='torch'):
    if backend == 'numpy':
        from torchstain.numpy.augmentors import NumpyMacenkoAugmentor
        return NumpyMacenkoAugmentor()
    elif backend == "torch":
        from torchstain.torch.augmentors import TorchMacenkoAugmentor
        return TorchMacenkoAugmentor()
    elif backend == "tensorflow":
        from torchstain.tf.augmentors import TensorFlowMacenkoAugmentor
        return TensorFlowMacenkoAugmentor()
    else:
        raise Exception(f'Unknown backend {backend}')
