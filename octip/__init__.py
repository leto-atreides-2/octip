"""
OCTIP initialization module.
"""

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'GPL-v3'
__version__ = '1.1'

from .parser import XMLParser
from .pre_processor import PreProcessor
from .retina_localizer import RetinaLocalizationDataset, RetinaLocalizer
from .utils import bscans_to_cscan
