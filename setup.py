import setuptools
from setuptools import setup

install_requires = [
    'torch>=1.10.0',
    'pytorch_ranger>=0.1.1',
]

setuptools.setup(
    name="oput",
    description=('pytorch-optimizer'),
    version='0.1.1',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires='>=3.6',
)
