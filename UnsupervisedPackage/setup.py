from setuptools import setup, find_packages

setup(
    name='unsupervised_ml',
    version='0.1.0',
    author='David Zapat',
    description='An unsupervised machine learning package',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn'
    ],
)
