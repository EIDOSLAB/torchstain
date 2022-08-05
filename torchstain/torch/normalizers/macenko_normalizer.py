from .numpy_macenko_normalizer import NumpyMacenkoNormalizer
from .torch_macenko_normalizer import TorchMacenkoNormalizer
from torchstain.tf.normalizers.tensorflow_macenko_normalizer import TensorFlowMacenkoNormalizer

def MacenkoNormalizer(backend='torch'):
    if backend == 'numpy':
        return NumpyMacenkoNormalizer()
    elif backend == "tensorflow":
    	return TensorFlowMacenkoNormalizer()
    elif backend == "torch":
    	return TorchMacenkoNormalizer()
    else:
    	raise Exception(f'Unknown backend {backend}')
