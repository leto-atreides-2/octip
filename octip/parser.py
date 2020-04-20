"""
OCTIP parsing module.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'Proprietary'
__version__ = '1.0'

import cv2
import os
from xml.etree import ElementTree


class XMLParser(object):
    """
    Parses on OCT volume in Heidelberg's XML format.
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
