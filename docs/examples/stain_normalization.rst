Stain Normalization
===================

The torchstain package supports three different backends: PyTorch,
TensorFlow, and NumPy. Below is a simple example of how to get started.
In this example we use PyTorch.

To run example, be sure to have installed the necessary dependencies:

``pip install torchstain[torch] torchvision opencv-python``

A simple usage example can be seen below:

.. code-block:: python

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

  normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
  normalizer.fit(T(target))

  t_to_transform = T(to_transform)
  norm, H, E = normalizer.normalize(I=t_to_transform, stains=True)


The generated result can be seen below:

.. image:: ../../data/result.png
  :alt: Stain normalized result


Different Backends
------------------

To use TensorFlow or NumPy backend, simply change the *backend*
argument to the *MacenkoNormalizer*. Also note that different for
different backends and normalization techniques, different
preprocessing may be required.

For TensorFlow instead perform:

.. code-block:: python

  import tensorflow as tf
  import numpy as np

  T = lambda x: tf.convert_to_tensor(np.moveaxis(x, -1, 0).astype("float32"))
  t_to_transform = T(to_transform)

Whereas for NumPy *no* preprocessing is required, given that the image
is already channel-last and uint8 dtype.
