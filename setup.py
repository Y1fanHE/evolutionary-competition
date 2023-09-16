import os
from setuptools import setup, find_packages


exec(open("evocomp/__init__.py").read())


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="evocomp",
    version=__version__,
    description="Evolutionary Competition",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Yifan He",
    author_email="heyif@outlook.com",
    license="MIT",
    url="https://github.com/Y1fanHE/evolutionary-competition",
    packages=find_packages(
        exclude=("img", "examples.*", "tests", "tests.*", "docs", "docs_source")
    ),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "deap",
        "matplotlib",
        "pygraphviz"
    ]
)
