from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="KerasModelManager",
    version="0.0.4",
    author="Daniel Mensing",
    author_email="daniel.mensing@gmx.net",
    description="A wrapper for Keras models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniel-men/ModelManager",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)