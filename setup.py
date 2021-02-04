import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pseudo_ood_generation",
    version="0.0.1",
    author="dunkelhaus",
    author_email="jena.suraj.k@gmail.com",
    description="Pseudo OOD Generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dunkelhaus/pseudo-ood-generation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
