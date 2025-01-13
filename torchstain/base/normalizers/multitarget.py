def MultiMacenkoNormalizer(backend="torch", **kwargs):
    if backend == "numpy":
        raise NotImplementedError("MultiMacenkoNormalizer is not implemented for NumPy backend")
    elif backend == "torch":
        from torchstain.torch.normalizers import TorchMultiMacenkoNormalizer
        return TorchMultiMacenkoNormalizer(**kwargs)
    elif backend == "tensorflow":
        raise NotImplementedError("MultiMacenkoNormalizer is not implemented for TensorFlow backend")
    else:
        raise Exception(f"Unsupported backend {backend}")
