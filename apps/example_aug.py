import cv2
import matplotlib.pyplot as plt
import torchstain
import torch
from torchvision import transforms
import time
import os


size = 1024
dir_path = os.path.dirname(os.path.abspath(__file__))
target = cv2.resize(cv2.cvtColor(cv2.imread(dir_path + "/../data/target.png"), cv2.COLOR_BGR2RGB), (size, size))
to_transform = cv2.resize(cv2.cvtColor(cv2.imread(dir_path + "/../data/source.png"), cv2.COLOR_BGR2RGB), (size, size))

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

t_to_transform = T(to_transform)

# setup augmentors for the different backends
augmentor = torchstain.augmentors.MacenkoAugmentor(backend='numpy')
augmentor.fit(to_transform)

tf_augmentor = torchstain.augmentors.MacenkoAugmentor(backend='tensorflow')
tf_augmentor.fit(t_to_transform)

torch_augmentor = torchstain.augmentors.MacenkoAugmentor(backend='torch')
torch_augmentor.fit(t_to_transform)


print("NUMPY" + "-"*20)

plt.figure()
plt.suptitle('numpy augmentor')
plt.subplot(4, 4, 1)
plt.title('Original')
plt.axis('off')
plt.imshow(to_transform)

for i in range(16):
    # generate augmented sample
    result = augmentor.augment()

    plt.subplot(4, 4, i + 1)
    if i == 1:
        plt.title('Augmented ->')
    plt.axis('off')
    plt.imshow(result)

plt.show()


print("TensorFlow (TF)" + "-"*20)

plt.figure()
plt.suptitle('tf augmentor')
plt.subplot(4, 4, 1)
plt.title('Original')
plt.axis('off')
plt.imshow(to_transform)

for i in range(16):
    # generate augmented sample
    result = tf_augmentor.augment()

    plt.subplot(4, 4, i + 1)
    if i == 1:
        plt.title('Augmented ->')
    plt.axis('off')
    plt.imshow(result)

plt.show()


print("Torch" + "-"*20)

plt.figure()
plt.suptitle('torch augmentor')
plt.subplot(4, 4, 1)
plt.title('Original')
plt.axis('off')
plt.imshow(to_transform)

for i in range(16):
    # generate augmented sample
    result = torch_augmentor.augment()

    plt.subplot(4, 4, i + 1)
    if i == 1:
        plt.title('Augmented ->')
    plt.axis('off')
    plt.imshow(result)

plt.show()
