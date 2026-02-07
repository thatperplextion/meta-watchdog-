"""
Meta-Watchdog: Self-Aware Machine Learning System
Setup configuration for package installation.
"""

from setuptools import setup, find_packages
import os

# Read the README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Meta-Watchdog: Self-Aware Machine Learning System"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    return []

setup(
    name="meta-watchdog",
    version="1.0.0",
    author="Meta-Watchdog Team",
    author_email="team@meta-watchdog.ai",
    description="Self-Aware Machine Learning System that predicts its own failures",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/meta-watchdog/meta-watchdog",
    project_urls={
        "Bug Tracker": "https://github.com/meta-watchdog/meta-watchdog/issues",
        "Documentation": "https://meta-watchdog.readthedocs.io/",
        "Source Code": "https://github.com/meta-watchdog/meta-watchdog",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.9,<3.14",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
        ],
        "all": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meta-watchdog=meta_watchdog.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "meta_watchdog": ["py.typed"],
    },
    zip_safe=False,
    keywords=[
        "machine-learning",
        "ai-safety",
        "self-aware-ai",
        "failure-prediction",
        "model-monitoring",
        "explainable-ai",
        "trustworthy-ml",
    ],
)
