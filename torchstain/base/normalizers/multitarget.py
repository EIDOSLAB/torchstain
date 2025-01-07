def MultiMacenkoNormalizer(backend='torch', **kwargs):
    if backend == 'torch':
        from torchstain.torch.normalizers.multitarget import MultiMacenkoNormalizer
        return MultiMacenkoNormalizer(**kwargs)
    else:
        raise Exception(f'Unsupported backend {backend}')
