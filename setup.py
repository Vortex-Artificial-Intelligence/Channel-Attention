import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="channel-attention",
    packages=setuptools.find_packages(),
    version="0.0.1",
    description="A plug-and-play channel attention mechanism module implemented in PyTorch.",  # 包的简短描述
    url="https://github.com/wwhenxuan/Channel-Attention",
    author="whenxuan",
    author_email="wwhenxuan@gmail.com",
    keywords=[
        "Deep Learning",
        "Neural Networks",
        "Attention Mechanism",
        "Channel Attention",
        "PyTorch",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
        "numpy",
    ],
)
