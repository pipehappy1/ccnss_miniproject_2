from setuptools import find_packages
from setuptools import setup

install_requires = [
    'numpy',
    'matplotlib',
    'scipy',
    'scikit-image'
]

tests_require = [
    'pytest',
]

setup(
    name="mini2",
    version="0.0.1",
    description="ccnss2018 mini 1",
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
)
