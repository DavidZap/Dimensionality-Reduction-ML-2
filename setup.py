from setuptools import setup, find_packages

dev_required = ["model","utils","pandas","numpy","matplotlib","scikit-learn"]

setup(
    name="Unsupervised_model",
    version="0.0.1",
    description="A companion to study dimensionality reduction",
    url="https://github.com/DavidZap/Dimensionality-Reduction-ML-2",
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    extras_require={"dev": dev_required},
    package_dir={"": "."},
)