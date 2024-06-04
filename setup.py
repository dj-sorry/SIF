from setuptools import setup, find_packages

setup(
    name='SIF',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'nltk',
        'scikit-learn',
        'torch',
        'pickle'
    ],
)
