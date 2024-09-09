"""
OCTIP utility module.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'GPL-v3'
__version__ = '1.1'

import cv2
import numpy as np
import os


def bscans_to_cscan(bscans,
                    image_directory = None,
                    file_extension = None):
    """
    Converts a dictionary or a list of B-scans to one C-scan.

    If image_directory is provided, B-scans are assumed to be image file names: images are then
    loaded from image_directory, optionally with a different file extension.

    :param bscans: dictionary or list of B-scans
    :param image_directory: optional image directory
    :param file_extension: optional file name extensions

    :return: the C-scan image as a slices x height x width Numpy array
    """
    assert isinstance(bscans, dict) or isinstance(bscans, list), \
        'Argument \'bscans\' must be a dictionary of a list!'
    bscan_list = bscans if isinstance(bscans, list) else sorted(bscans)
    if image_directory is not None:
        assert os.path.exists(image_directory), \
            'Directory \'{}\' does not exist!'.format(image_directory)
        file_names = bscan_list
        bscan_list = []
        for file_name in file_names:
            extension = file_extension or os.path.splitext(file_name)[1]
            file_name = os.path.join(image_directory, extract_base_name(file_name) + extension)
            assert os.path.exists(file_name), 'File \'{}\' does not exist!'.format(file_name)
            bscan_list.append(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE))
    return np.asarray(bscan_list)


def extract_base_name(file_name):
    """
    Base name of a file, without extension (/path/to/file/base_name.ext -> base_name).

    :param file_name: input file name

    :return: the base name without extension
    """
    return os.path.splitext(os.path.split(file_name)[1])[0]


def to_list(element):
    """
    Converts a tuple, a scalar (or a list) to a list.

    :param element: input element

    :return: the converted element
    """
    if isinstance(element, tuple):
        element = list(element)
    elif not isinstance(element, list):
        element = [element]
    return element
