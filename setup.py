#   encoding: utf8
#   setup.py

from pip._internal.req import parse_requirements
from setuptools import setup, find_packages


def load_requirements(fname):
    return [str(r.req) for r in parse_requirements(fname, session='')]


setup(name='quffka',
      version='0.1.0',
      packages=find_packages(),
      install_requires=load_requirements('requirements.txt'),
      entry_points={
          'console_scripts': [
              'cook-datasets = quffka.cook_datasets:main',
              'approximate-kernels = quffka.run:main',
              'measure-time = quffka.time:main',
          ]
      })
