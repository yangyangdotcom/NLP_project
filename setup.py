# setup.py

from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = ["mkdocs==1.3.0", "mkdocstrings==0.18.1"]

style_packages = ["black==22.3.0", "flake8==3.9.2", "isort==5.10.1"]

test_packages = [
    "pytest==7.1.2",
    "pytest-cov==2.10.1",
    "great-expectations==0.15.15"
]
setup(
    name="bnai",
    version=0.1,
    description="Classify machine learning projects",
    author="Benjamin Tan",
    author_email="tanciyoung99@gmail.com",
    url="https://madewithml.com/",
    python_requires=">=3.7",
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages + style_packages + test_packages,
        "docs": docs_packages,
        "test": test_packages,
    },
)
