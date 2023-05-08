def MacenkoAugmentor(backend='torch'):
    if backend == 'numpy':
        from torchstain.numpy.augmentors import NumpyMacenkoAugmentor
        return NumpyMacenkoAugmentor()
    elif backend == "torch":
        raise NotImplementedError()
    elif backend == "tensorflow":
        from torchstain.tf.augmentors import TensorFlowMacenkoAugmentor
        return TensorFlowMacenkoAugmentor()
    else:
        raise Exception(f'Unknown backend {backend}')
