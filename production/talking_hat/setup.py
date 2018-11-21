import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="talking_hat",
    version="0.0.1",
    author="Godric Gryffindor",
    author_email="godric@example.com",
    description="A talking hat module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
