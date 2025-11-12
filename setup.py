import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-pruning",
    version="v1.5.1",
    author="Avraham Raviv",
    description="Towards Any Structural Pruning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://gitlab-srv/avrahamra/sirc_torch_pruning.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)