"""
OCTIP parsing module.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'Proprietary'
__version__ = '1.1'

import cv2
import glob
import numpy as np
import os
import pydicom
from collections import defaultdict
from enum import Enum
from xml.etree import ElementTree


class PLEXEliteFileType(Enum):
    """
    PLEXElite dataset file types.
    """
    STRUCTURE_CUBE = 0
    FLOW_CUBE = 1
    RETRACE_OCT_FRAMES = 2
    NOISE_FRAME = 3
    LSO_IMAGE = 4
    IRIS_IMAGE = 5
    ENFACE_IMAGE = 6
    ANALYSIS_FILE = 7


class PLEXEliteParser(object):
    """
    Parses a PLEXElite eye exam in DICOM format.
    """

    _file_type = {'1.2.276.0.75.2.2.44.6': PLEXEliteFileType.STRUCTURE_CUBE,
                  '1.2.276.0.75.2.2.44.15': PLEXEliteFileType.FLOW_CUBE,
                  '1.2.276.0.75.2.2.44.4': PLEXEliteFileType.RETRACE_OCT_FRAMES,
                  '1.2.276.0.75.2.2.44.3': PLEXEliteFileType.NOISE_FRAME,
                  '1.2.276.0.75.2.2.44.1': PLEXEliteFileType.LSO_IMAGE,
                  '1.2.276.0.75.2.2.44.2': PLEXEliteFileType.IRIS_IMAGE,
                  '1.2.276.0.75.2.2.44.10': PLEXEliteFileType.ENFACE_IMAGE}

    def __init__(self,
                 directory):
        """
        PLEXEliteParser constructor.

        :param directory: directory containing the DICOM files.
        """
        dicoms = glob.glob(os.path.join(directory, '*.DCM'))
        self.datasets = defaultdict(list)
        for dicom in dicoms:
            dataset = pydicom.dcmread(dicom)
            if PLEXEliteParser.__check_plex_elite(dataset):
                file_type = PLEXEliteParser.__file_type(dataset)
                self.datasets[file_type].append(dataset)

    @staticmethod
    def __check_plex_elite(dataset):
        """
        Checks that a dataset was acquired by a PLEXElite.

        :param dataset: image dataset

        :return: whether or not the dataset was acquired by a PLEXElite
        """
        sop_instance_uid = dataset[0x8, 0x18].value
        if isinstance(sop_instance_uid, list) and len(sop_instance_uid) > 0:
            sop_instance_uid = sop_instance_uid[0]
        if isinstance(sop_instance_uid, str):
            return sop_instance_uid.startswith('1.2.276.0.75.2.2.44')
        else:
            return False

    @staticmethod
    def __dimension(dataset,
                    group,
                    element):
        """
        Gets one dimension of an image dataset identified by the (group, element) tag.

        :param dataset: image dataset
        :param group: tag group
        :param element: tag element in the group

        :return: the value of the dimension (1 if not found)
        """
        try:
            value = dataset[group, element].value
            if isinstance(value, str):
                value = ''.join(filter(str.isdigit, value))
            value = int(value)
            return value
        except Exception:
            return 1

    @classmethod
    def __file_type(cls,
                    dataset):
        """
        Gets the file type of a dataset.

        :param dataset: image dataset

        :return: the file type (PLEXEliteFileType)
        """
        czm_iod_uid_file_type = dataset[0x57, 0x1].value
        if isinstance(czm_iod_uid_file_type, list) and len(czm_iod_uid_file_type) > 0:
            czm_iod_uid_file_type = czm_iod_uid_file_type[0]
        if isinstance(czm_iod_uid_file_type, str):
            for key in cls._file_type:
                if czm_iod_uid_file_type.startswith(key):
                    return cls._file_type[key]
        return PLEXEliteFileType.ANALYSIS_FILE

    def load_images(self,
                    file_type):
        """
        Loads images from datasets with a given file type.

        :param file_type: the file type (PLEXEliteFileType)

        :return: a list of images
        """
        images = []
        if file_type in self.datasets:
            datasets = self.datasets[file_type]
            for dataset in datasets:

                # size of the data ([NumberOfFrames, Rows, Columns])
                dimensions = [PLEXEliteParser.__dimension(dataset, 0x28, 0x8)] + \
                             [PLEXEliteParser.__dimension(dataset, 0x28, 0x10)] + \
                             [PLEXEliteParser.__dimension(dataset, 0x28, 0x11)]
                num_dimensions = 0
                for d in dimensions:
                    if d > 1:
                        num_dimensions += 1
                theoretical_size = np.prod(dimensions)

                if num_dimensions == 2:

                    # cloning the 2-D array
                    images.append(np.copy(dataset.pixel_array))

                elif num_dimensions == 3:

                    # pixel data
                    pixel_data = dataset[0x7fe0, 0x10].value
                    actual_size = len(pixel_data)

                    # padding info
                    padding = actual_size - theoretical_size
                    padding_by_frame = int(padding / (dimensions[0] + 1))

                    # forming the 3-D array
                    frame_size = dimensions[1] * dimensions[2]
                    frames = []
                    for num_frame in range(dimensions[0]):
                        data_ptr = padding_by_frame * (num_frame + 1) + frame_size * num_frame
                        data = pixel_data[data_ptr: data_ptr + frame_size]
                        frames.append(np.reshape(np.frombuffer(data, dtype = np.uint8),
                                                 [dimensions[1], dimensions[2]]))
                    images.append(np.array(frames))

        return images


class XMLParser(object):
    """
    Parses an OCT volume in Heidelberg's XML format.
    """

    def __init__(self,
                 url,
                 load_images = False):
        """
        XMLParser constructor.

        :param url: URL of the XML file
        :param load_images: should images be loaded?
        """
        self.url = url
        self.bscans = dict()  # B-scans (file names or data) indexed by their localization
        self.laterality = ''
        self.localizer = None
        self.localizer_scale = None
        self.images_loaded = load_images

        # loop over B-scans
        tree = ElementTree.parse(url)
        root = tree.getroot().find('BODY').find('Patient').find('Study').find('Series')
        directory = os.path.dirname(url)
        for image_node in root.findall('Image'):

            # image data
            image_data_node = image_node.find('ImageData')
            _, image_url = image_data_node.find('ExamURL').text.rsplit('\\', 1)
            url = os.path.join(directory, image_url)
            image = cv2.imread(url, cv2.IMREAD_GRAYSCALE) if load_images else url

            # image metadata
            self.laterality = image_node.find('Laterality').text
            image_type = image_node.find('ImageType').find('Type').text

            # coordinate data
            context_node = image_node.find('OphthalmicAcquisitionContext')
            if image_type == 'LOCALIZER':
                self.localizer = image
                self.localizer_scale = (float(context_node.find('ScaleX').text),
                                        float(context_node.find('ScaleY').text))
            elif image_type == 'OCT':
                start_node = context_node.find('Start').find('Coord')
                start = (float(start_node.find('X').text), float(start_node.find('Y').text))
                end_node = context_node.find('End').find('Coord')
                end = (float(end_node.find('X').text), float(end_node.find('Y').text))
                self.bscans[(start, end)] = image

        # scaling the localizations
        if self.localizer is not None:
            bscans_scaled = dict()
            scale_x = self.localizer_scale[0]
            scale_y = self.localizer_scale[1]
            for (start, end) in self.bscans:
                start_scaled = (start[0] / scale_x, start[1] / scale_y)
                end_scaled = (end[0] / scale_x, end[1] / scale_y)
                bscans_scaled[(start_scaled, end_scaled)] = self.bscans[(start, end)]
            self.bscans = bscans_scaled

    def sorted_bscans(self):
        """
        Returns the B-scans sorted according to their localizations.

        If load_images = False, the B-scan file names are returned, otherwise the data is returned.

        :return: list of B-scans sorted according to their localizations
        """
        return [self.bscans[localization] for localization in sorted(self.bscans)]

    @staticmethod
    def study_date(url):
        """
        Returns the study date.

        :param url: URL of the XML file

        :return: the study date in 'year-month-day' format (e.g. '2020-03-20')
        """
        tree = ElementTree.parse(url)
        root = tree.getroot().find('BODY').find('Patient').find('Study').find('StudyDate')\
            .find('Date')
        year = root.find('Year').text
        month = root.find('Month').text
        if len(month) == 1:
            month = '0' + month
        day = root.find('Day').text
        if len(day) == 1:
            day = '0' + day
        return year + '-' + month + '-' + day
