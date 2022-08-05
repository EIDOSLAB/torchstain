__version__ = '1.1.0'
#__BACKEND__ = 'torch'

#import os
#os.environ['backend'] = "torch"

from torchstain.base.normalizers.macenko_normalizer import MacenkoNormalizer

try:
    import torch
    from torchstain.torch_backend.normalizers.torch_macenko_normalizer import TorchMacenkoNormalizer
except ModuleNotFoundError:
    try:
        import tensorflow
        from torchstain.tf_backend.normalizers.tensorflow_macenko_normalizer import TensorFlowMacenkoNormalizer
    except ModuleNotFoundError:
        pass
