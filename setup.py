"""
OCTIP setup module.
"""

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'GPL-v3'
__version__ = '1.1'

from setuptools import setup

setup(name = 'octip',
      version = '1.0',
      license = 'Proprietary',
      description = 'OCT Internship Package',
      author = 'Gwenole Quellec',
      author_email = 'gwenole.quellec@inserm.fr',
      packages = ['octip'],
      scripts = ['octip-dataset-split.py', 'octip-spectralis-2-nifti.py'],
      install_requires = ['nibabel', 'numpy', 'opencv-python', 'pandas', 'scikit-learn',
                          'scipy', 'segmentation-models'])
