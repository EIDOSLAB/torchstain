import pathlib
from setuptools import setup, find_packages, find_namespace_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name='torchstain',
    version='1.1.0',
    description='Pytorch stain normalization utils',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/EIDOSlab/torchstain',
    author='EIDOSlab',
    author_email='eidoslab@di.unito.it',
    license='MIT',
    #package_dir={
    #            'torchstain': 'torchstain',
    #            'torchstain.torch': 'torchstain/torch',
    #            'torchstain.tf': 'torchstain/tf'},
    #packages=['torchstain', 'torchstain.torch',],
    #          #'torchstain.tf'],
    packages=find_packages(exclude=('tests')),
    #packages=find_namespace_packages(where='torchstain'),
    #packages=["torchstain", "torchstain.torch"],
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
