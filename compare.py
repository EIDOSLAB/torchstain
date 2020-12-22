import torch
from torchvision import transforms
import torchstain

import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from tabulate import tabulate

def measure(size, N):
    print('=> Measuring time for size', size)

    target = cv2.resize(cv2.cvtColor(cv2.imread("./data/target.png"), cv2.COLOR_BGR2RGB), (size, size))
    to_transform = cv2.resize(cv2.cvtColor(cv2.imread("./data/source.png"), cv2.COLOR_BGR2RGB), (size, size))

    normalizer = torchstain.MacenkoNormalizer(backend='numpy')
    normalizer.fit(target)


    T = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])

    torch_normalizer = torchstain.MacenkoNormalizer(backend='torch')
    torch_normalizer.fit(T(target))

    t_to_transform = T(to_transform)

    t_np = []
    start_np = time.perf_counter()
    for i in range(N):
        tic = time.perf_counter()
        _ = normalizer.normalize(to_transform)
        toc = time.perf_counter()
        t_np.append(toc-tic)
    end_np = time.perf_counter()
    t_np = np.array(t_np)


    t_torch = []
    start_torch = time.perf_counter()
    for i in range(N):
        tic = time.perf_counter()
        _ = torch_normalizer.normalize(t_to_transform)
        toc = time.perf_counter()
        t_torch.append(toc-tic)
    end_torch = time.perf_counter()
    t_torch = np.array(t_torch)

    """
    print(f'Results of {N} runs:')
    print(f'numpy: {t_np.mean():.4f}s ± {t_np.std():.4f} (tot: {end_np-start_np:.4f}s)')
    print(f'torch: {t_torch.mean():.4f}s ± {t_torch.std():.4f} (tot: {end_torch-start_torch:.4f}s)')
    """

    return t_np, end_np-start_np, t_torch, end_torch-start_torch

table = []
for size in [224, 448, 672, 896, 1120, 1344, 1568, 1792]:
    t_np, tot_np, t_torch, tot_torch = measure(size, N=10)
    row = [size, f'{t_np.mean():.4f}s ± {t_np.std():.4f}', f'{tot_np:.4f}s', f'{t_torch.mean():.4f}s ± {t_torch.std():.4f}', f'{tot_torch:.4f}s']
    table.append(row)

print(tabulate(table, headers=['size', 'numpy avg. time', 'numpy tot. time', 'torch avg. time', 'torch tot. time'], tablefmt='github'))
