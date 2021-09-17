import cv2
import matplotlib.pyplot as plt
import torchstain
import torch
from torchvision import transforms
import time


size = 1024
target = cv2.resize(cv2.cvtColor(cv2.imread("./data/target.png"), cv2.COLOR_BGR2RGB), (size, size))
to_transform = cv2.resize(cv2.cvtColor(cv2.imread("./data/source.png"), cv2.COLOR_BGR2RGB), (size, size))

normalizer = torchstain.MacenkoNormalizer(backend='numpy')
normalizer.fit(target)

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

torch_normalizer = torchstain.MacenkoNormalizer(backend='torch')
torch_normalizer.fit(T(target))

tf_normalizer = torchstain.MacenkoNormalizer(backend='tensorflow')
tf_normalizer.fit(T(target))

t_to_transform = T(to_transform)

t_ = time.time()
norm, H, E = normalizer.normalize(I=to_transform, stains=True)
print("numpy runtime:", time.time() - t_)

plt.figure()
plt.suptitle('numpy normalizer')
plt.subplot(2, 2, 1)
plt.title('Original')
plt.axis('off')
plt.imshow(to_transform)

plt.subplot(2, 2, 2)
plt.title('Normalized')
plt.axis('off')
plt.imshow(norm)

plt.subplot(2, 2, 3)
plt.title('H')
plt.axis('off')
plt.imshow(H)

plt.subplot(2, 2, 4)
plt.title('E')
plt.axis('off')
plt.imshow(E)
plt.show()

t_ = time.time()
norm, H, E = torch_normalizer.normalize(I=t_to_transform, stains=True)
print("torch runtime:", time.time() - t_)

plt.figure()
plt.suptitle('torch normalizer')
plt.subplot(2, 2, 1)
plt.title('Original')
plt.axis('off')
plt.imshow(to_transform)

plt.subplot(2, 2, 2)
plt.title('Normalized')
plt.axis('off')
plt.imshow(norm)

plt.subplot(2, 2, 3)
plt.title('H')
plt.axis('off')
plt.imshow(H)

plt.subplot(2, 2, 4)
plt.title('E')
plt.axis('off')
plt.imshow(E)
plt.show()

t_ = time.time()
norm, H, E = tf_normalizer.normalize(I=t_to_transform, stains=True)
print("tf runtime:", time.time() - t_)

plt.figure()
plt.suptitle('tensorflow normalizer')
plt.subplot(2, 2, 1)
plt.title('Original')
plt.axis('off')
plt.imshow(to_transform)

plt.subplot(2, 2, 2)
plt.title('Normalized')
plt.axis('off')
plt.imshow(norm)

plt.subplot(2, 2, 3)
plt.title('H')
plt.axis('off')
plt.imshow(H)

plt.subplot(2, 2, 4)
plt.title('E')
plt.axis('off')
plt.imshow(E)
plt.show()
