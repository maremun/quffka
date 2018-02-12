#   encoding: utf8
#   setup.py

from setuptools import setup, find_packages


setup(name='quffka',
      version='0.1.0',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'cook-datasets = quffka.cook_datasets:main',
              'approximate-kernels = quffka.run:main',
              'measure-time = quffka.time:main',
          ]
      })
