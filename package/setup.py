from setuptools import setup, find_packages

setup(
    name="gdig",
    version="0.1.0",
    description="G-DIG: Gradient-based DIverse and hiGh-quality data selection for diagnosing model performance",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/G-DIG",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "tqdm",
        "numpy",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'gdig=gdig.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 