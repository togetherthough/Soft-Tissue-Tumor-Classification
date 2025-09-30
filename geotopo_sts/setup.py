"""
Setup script for GeoTopo-STS package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README with UTF-8 encoding
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

# Read requirements with UTF-8 encoding
req_file = Path(__file__).parent / "requirements.txt"
requirements = []
if req_file.exists():
    with open(req_file, encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="geotopo-sts",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Geometry-Topology Soft Tissue Sarcoma Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/geotopo-sts",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "pyyaml>=6.0",
        "nibabel>=5.0.0",
        "trimesh>=3.23.0",
        "torch-geometric>=2.3.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "full": [
            "SimpleITK>=2.2.0",
            "gudhi>=3.8.0",
            "geoopt>=0.5.0",
            "e3nn>=0.5.0",
            "pymeshlab>=2022.2",
            "wandb>=0.15.0",
            "umap-learn>=0.5.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "geotopo-preprocess=geotopo_sts.preprocess_pipeline:main",
            "geotopo-train=geotopo_sts.train:main",
            "geotopo-eval=geotopo_sts.eval:main",
        ]
    },
    include_package_data=True,
    package_data={
        "geotopo_sts": ["config.yaml", "*.md"]
    },
    zip_safe=False,
)
