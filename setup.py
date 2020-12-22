from setuptools import setup

setup(
    name='torchstain',
    version='1.0',
    description='pytorch stain normalization utils',
    url='git@github.com:EIDOSlab/torchstain.git',
    author='Carlo Alberto Barbano',
    author_email='carlo.alberto.barbano@outlook.com',
    license='MIT',
    packages=['torchstain'],
    zip_safe=False,
    install_requires=[
        'torch',
        'numpy'
    ]
)
