:github_url: https://github.com/EIDOSLAB/torchstain/tree/main/docs

TorchStain
-------------------

TorchStain is a modular Python package for GPU-accelerated stain normalization
and augmentation for histopathological image analysis. It supports PyTorch,
TensorFlow, and NumPy backends.

Installation
------------

The latest release of TorchStain can be installed from
`PyPI <https://pypi.org/project/torchstain/>`_ using

``pip install torchstain``

To install a specific backend use either torchstain[torch] or torchstain[tf].
The numpy backend is included by default in both.

You may also install directly from GitHub, using the following command:

``pip install git+https://github.com/EIDOSLAB/torchstain``

.. toctree::
   :glob:
   :caption: Background
   :maxdepth: 2

   background/*

.. toctree::
   :glob:
   :caption: Examples
   :maxdepth: 2

   examples/*

.. toctree::
   :glob:
   :caption: Frequently Asked Questions
   :maxdepth: 2

   faq/*


.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api

The Team
--------

The development of TorchStain is led by researchers at [EIDOSLAB](https://eidos.di.unito.it/)
and [SINTEF MIA](https://www.sintef.no/en/expertise/sintef-technology-and-society/medical-technology/).
We are also very grateful to the open source community for contributing ideas, bug fixes, and issues.

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/EIDOSLAB/torchstain/blob/main/LICENSEissues>`_.


License
-------

TorchStain is licensed under the `MIT License <https://github.com/EIDOSLAB/torchstain/blob/main/LICENSE>`_.


Indices and Tables
==================

* :ref:`genindex`