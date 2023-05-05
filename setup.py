import pathlib
from setuptools import setup, find_packages, find_namespace_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='torchstain',
    version='1.3.0',
    description='Stain normalization tools for histological analysis and computational pathology',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/EIDOSlab/torchstain',
    author='Carlo Alberto Barbano, André Pedersen',
    author_email='eidoslab@di.unito.it',
    license='MIT',
    packages=find_packages(exclude=('tests')),
    zip_safe=False,
    install_requires=[
        'numpy'
    ],
    extras_require={
        "tf": ["tensorflow"],
        "torch": ["torch"],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)
