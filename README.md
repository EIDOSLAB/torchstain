# torchstain

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![tests](https://github.com/EIDOSLAB/torchstain/workflows/tests/badge.svg)](https://github.com/EIDOSLAB/torchstain/actions)
[![Pip Downloads](https://img.shields.io/pypi/dm/torchstain?label=pip%20downloads&logo=python)](https://pypi.org/project/torchstain/)
[![DOI](https://zenodo.org/badge/323590093.svg)](https://zenodo.org/badge/latestdoi/323590093)

GPU-accelerated stain normalization tools for histopathological images. Compatible with PyTorch, TensorFlow, and Numpy.
Normalization algorithms currently implemented:

- Macenko [\[1\]](#reference) (ported from [numpy implementation](https://github.com/schaugf/HEnorm_python))
- Reinhard [\[2\]](#reference)
- Modified Reinhard [\[3\]](#reference)

## Installation

```bash
pip install torchstain
```

To install a specific backend use either ```torchstain[torch]``` or ```torchstain[tf]```. The numpy backend is included by default in both.

## Example Usage

```python
import torch
from torchvision import transforms
import torchstain
import cv2

target = cv2.cvtColor(cv2.imread("./data/target.png"), cv2.COLOR_BGR2RGB)
to_transform = cv2.cvtColor(cv2.imread("./data/source.png"), cv2.COLOR_BGR2RGB)

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
torch_normalizer.fit(T(target))

t_to_transform = T(to_transform)
norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)
```

![alt text](data/result.png)

## Implemented algorithms

| Algorithm | numpy | torch | tensorflow |
|-|-|-|-|
| Macenko | &check; | &check; | &check; |
| Reinhard | &check; | &check; | &check; |
| Modified Reinhard | &check; | &check; | &check; |

## Backend comparison

Results with 10 runs per size on a Intel(R) Core(TM) i5-8365U CPU @ 1.60GHz

|   size | numpy avg. time   | torch avg. time   | tf avg. time     |
|--------|-------------------|-------------------|------------------|
|    224 | 0.0182s ± 0.0016  | 0.0180s ± 0.0390  | 0.0048s ± 0.0002 |
|    448 | 0.0880s ± 0.0224  | 0.0283s ± 0.0172  | 0.0210s ± 0.0025 |
|    672 | 0.1810s ± 0.0139  | 0.0463s ± 0.0301  | 0.0354s ± 0.0018 |
|    896 | 0.3013s ± 0.0377  | 0.0820s ± 0.0329  | 0.0713s ± 0.0008 |
|   1120 | 0.4694s ± 0.0350  | 0.1321s ± 0.0237  | 0.1036s ± 0.0042 |
|   1344 | 0.6640s ± 0.0553  | 0.1665s ± 0.0026  | 0.1663s ± 0.0021 |
|   1568 | 1.1935s ± 0.0739  | 0.2590s ± 0.0088  | 0.2531s ± 0.0031 |
|   1792 | 1.4523s ± 0.0207  | 0.3402s ± 0.0114  | 0.3080s ± 0.0188 |

## Reference

- [1] Macenko, Marc et al. "A method for normalizing histology slides for quantitative analysis." 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2009.
- [2] Reinhard, Erik et al. "Color transfer between images." IEEE Computer Graphics and Applications. IEEE, 2001.
- [3] Roy, Santanu et al. "Modified Reinhard Algorithm for Color Normalization of Colorectal Cancer Histopathology Images". 2021 29th European Signal Processing Conference (EUSIPCO), IEEE, 2021.

## Citing

If you find this software useful for your research, please cite it as: 

```bibtex
@software{barbano2022torchstain,
  author       = {Carlo Alberto Barbano and
                  André Pedersen},
  title        = {EIDOSLAB/torchstain: v1.2.0-stable},
  month        = aug,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v1.2.0-stable},
  doi          = {10.5281/zenodo.6979540},
  url          = {https://doi.org/10.5281/zenodo.6979540}
}
```

Torchstain was originally developed within the [UNITOPATHO](https://github.com/EIDOSLAB/UNITOPATHO) data collection, which you can cite as:

```bibtex
@inproceedings{barbano2021unitopatho,
  title={UniToPatho, a labeled histopathological dataset for colorectal polyps classification and adenoma dysplasia grading},
  author={Barbano, Carlo Alberto and Perlo, Daniele and Tartaglione, Enzo and Fiandrotti, Attilio and Bertero, Luca and Cassoni, Paola and Grangetto, Marco},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={76--80},
  year={2021},
  organization={IEEE}
}
```
