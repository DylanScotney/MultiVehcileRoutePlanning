
from setuptools import setup, find_packages

setup(
    name="Python mVRP",
    author="Dylan Scotney",
    version="0.0.1",
    description="Library for multi vehicle route planning",
    packages=find_packages(exclude=['*tests', '*testing']),
    install_requires=[],
    tests_require=[]
)
