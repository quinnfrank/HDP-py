import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hdp_py",
    version="0.1",
    author="Quinn Frank, Morris Greenberg, George Lindner",
    author_email="quinn.frank@duke.edu, morris.greenberg@duke.edu, george.lindner@duke.edu",
    description="A Python implementation of the hierarchical Dirichlet process proposed in Teh, et al (2006)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quinnfrank/hdp-py",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
    python_requires='>=3.7',
)