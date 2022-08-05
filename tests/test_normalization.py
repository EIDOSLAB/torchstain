import os
import cv2
import matplotlib.pyplot as plt
import torchstain
import torch
from torchvision import transforms
import time
from skimage.metrics import structural_similarity as ssim


size = 1024
curr_file_path = os.path.dirname(os.path.realpath(__file__))
print("dir path:", curr_file_path)
target = cv2.resize(cv2.cvtColor(cv2.imread(curr_file_path + "../data/target.png"), cv2.COLOR_BGR2RGB), (size, size))
to_transform = cv2.resize(cv2.cvtColor(cv2.imread(curr_file_path + "../data/source.png"), cv2.COLOR_BGR2RGB), (size, size))

# setup preprocessing and preprocess image to be normalized
T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])
t_to_transform = T(to_transform)

# initialize normalizers for each backend and fit to target image
normalizer = torchstain.MacenkoNormalizer(backend='numpy')
normalizer.fit(target)

torch_normalizer = torchstain.MacenkoNormalizer(backend='torch')
torch_normalizer.fit(T(target))

tf_normalizer = torchstain.MacenkoNormalizer(backend='tensorflow')
tf_normalizer.fit(T(target))

# transform
result_numpy, _, _ = normalizer.normalize(I=to_transform, stains=True)
result_torch, _, _ = torch_normalizer.normalize(I=t_to_transform, stains=True)
result_tf, _, _ = tf_normalizer.normalize(I=t_to_transform, stains=True)

# calculate SSIM to use as metric for assessment if the results are similar
print(ssim(result_numpy, result_torch))
print(ssim(result_numpy, result_tf))

# assess whether the normalized images are identical across backends
np.testing.assert_equal(result_numpy, result_torch, verbose=True)
np.testing.assert_equal(result_numpy, result_tf, verbose=True)
