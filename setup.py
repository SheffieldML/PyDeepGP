#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
import numpy

# Version number
version = '1.0'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'DGP',
      version = version,
      author = read('AUTHORS.txt'),
      author_email = "z.dai@sheffield.ac.uk",
      description = ("Deep Gaussian Process"),
      license = "BSD 3-clause",
      keywords = "deep learning",
      url = "",
      packages = ["deepgp",
                  "deepgp.inference",
                  "deepgp.models"],
      package_dir={'deepgp': 'deepgp'},
      py_modules = ['deepgp.__init__'],
      long_description=read('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.12','GPy>=1.0'],
      include_dirs=[numpy.get_include()],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5'
                   ]
      )