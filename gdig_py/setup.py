from setuptools import setup, find_packages

setup(
    name="gdig-py",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "nngeometry>=0.0.4",
    ],
    python_requires=">=3.8",
    author="Xingyuan Pan",
    author_email="xingyuan.pan@example.com",
    description="G-DIG: Gradient-based Diverse and high-quality Instruction data selection for Machine Translation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gdig-py",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 