from setuptools import setup, find_packages

setup(
    name='torchdr',
    version='0',
    description='Torch Dimensionality Reduction Library',

    # The project's main homepage.
    url='https://github.com/TorchDR/TorchDR',
    download_url='https://github.com/TorchDR/TorchDR/archive/refs/tags/v0.0.0-alpha.tar.gz',

    # Author details
    author='Hugues Van Assel, TorchDR contributors',
    author_email='vanasselhugues@gmail.com',

    # Choose your license
    license='BSD 3-Clause',
    # What does your project relate to?
    keywords='dimensionality reduction',

    packages=find_packages(),
)