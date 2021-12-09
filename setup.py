from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="KerasModelManager",
    version="0.0.10",
    author="Daniel Mensing",
    author_email="daniel.mensing@gmx.net",
    description="A wrapper for Keras models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniel-men/ModelManager",
    download_url = 'https://github.com/daniel-men/ModelManager/archive/0.0.6.tar.gz',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'dill',
        'tensorflow',
        'python_utility_functions'
    ]
)