#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

# Version number
version = '1.0'

def read(fname):
    with open(fname) as f:
        contents = f.read()
    return contents

setup(name = 'DGP',
      version = version,
      author = read('AUTHORS.txt'),
      author_email = "damianou@amazon.com",
      description = ("Deep Gaussian Process"),
      license = "BSD 3-clause",
      keywords = "deep learning",
      url = "",
      packages = ["deepgp",
                  "deepgp.encoder",
                  "deepgp.inference",
                  "deepgp.layers",
                  "deepgp.util",
                  "deepgp.models",
                  "deepgp.testing"],
      package_dir={'deepgp': 'deepgp'},
      package_data={'deepgp.testing': ['*.txt', '*.h5']},
      include_package_data=True,
      py_modules = ['deepgp.__init__'],
      long_description=read('README.md'),
      python_requires='>=3.10',
      install_requires=[
          'numpy>=1.7',
          'scipy>=0.12',
          'GPy>=1.0',
      ],
      extras_require={
          'test': [
              'matplotlib',
              'h5py',
              'tables',
          ],
      },
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.10',
                   'Programming Language :: Python :: 3.11',
                   'Programming Language :: Python :: 3.12',
                   ]
      )
