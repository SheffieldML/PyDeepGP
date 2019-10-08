#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
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
                  "deepgp.inference",
                  "deepgp.layers",
                  "deepgp.util",
                  "deepgp.models"],
      package_dir={'deepgp': 'deepgp'},
      py_modules = ['deepgp.__init__'],
      long_description=read('README.md'),
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
              'theano',
          ],
      },
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7'
                   ]
      )
