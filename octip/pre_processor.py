"""
OCTIP pre-processing module.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'Proprietary'
__version__ = '1.1'

import cv2
import numpy as np
import os
import tensorflow as tf
from scipy import ndimage
from sklearn import linear_model
from .utils import extract_base_name, to_list


class PreProcessor(object):
    """
    Pre-processes OCT volumes.

    The following operations are performed:
      * the retinal surface is flattened,
      * the retina and the top layers of the choroid are cropped,
      * low-quality A-scans are discarded,
      * optionally, intensity is normalized.
    """

    TOP_INTENSITY = 200

    def __init__(self,
                 max_height,
                 min_height = None,
                 image_size = None,
                 min_distance_mask = 3,
                 cutoff_segmentation = .5,
                 sigma = 1.,
                 max_diff_tops = 5,
                 normalize_intensities = True):
        """
        PreProcessor constructor.

        :param max_height: maximum height of the A-scan selections
        :param min_height: minimum valid height of the A-scan selections (= max_height by default)
        :param image_size: output image size (no resizing by default)
        :param min_distance_mask: minimum distance between the retina and the mask
        :param cutoff_segmentation: cutoff for binarizing retina segmentations (in [0;1])
        :param sigma: standard deviation for the A-scan alignment Gaussian filter
        :param max_diff_tops: maximum difference between retina top estimations provided by various
                              retina segmentations
        :param normalize_intensities: should intensities in A-scans be normalized?
        """
        assert 0 <= cutoff_segmentation <= 1, 'Argument cutoff_segmentation must be in [0;1].'
        self.__max_height = max_height
        self.__min_height = min(max_height, min_height or max_height)
        self.__image_size = image_size
        self.__min_distance_mask = min_distance_mask
        self.__cutoff_segmentation = cutoff_segmentation
        self.__sigma = sigma
        self.__max_diff_tops = max_diff_tops
        self.__normalize_intensities = normalize_intensities

    def __call__(self,
                 samples,
                 segmentation_directories,
                 output_directory):
        """
        Pre-processes a list of B-scans and saves the result in a directory.

        Retinal segmentations are used for pre-processing. For improved performance, multiple
        retinal segmentation models can be used: one directory must be provided per segmentation
        model.

        :param samples: list of images to process (absolute or relative path to image files)
        :param segmentation_directories: list of directories where retinal segmentations are saved
        :param output_directory: output image directory
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        progress = tf.keras.utils.Progbar(len(samples), unit_name = 'b-scans')
        for bscan in samples:
            base_name = extract_base_name(bscan)
            segmentations = [os.path.join(directory, base_name + '.png') for directory in
                             segmentation_directories]
            preprocessed = self.process_image(
                cv2.imread(bscan, cv2.IMREAD_GRAYSCALE),
                [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in segmentations])
            cv2.imwrite(os.path.join(output_directory, base_name + '.png'), preprocessed)
            progress.add(1)

    @staticmethod
    def __compute_mask(image):
        """
        Computes a mask for the informative part of the image.

        :param image: input image

        :return: the binary mask
        """
        mask = np.ones_like(image)
        height, width = image.shape

        # loop on columns in a row
        def loop_j(i, range_j):
            for j in range_j:
                if image[i, j] == 0:
                    mask[i, j] = 0
                else:
                    break

        # loop on rows in a column
        def loop_i(range_i, j):
            for i in range_i:
                if image[i, j] == 0:
                    mask[i, j] = 0
                else:
                    break

        for i in range(height):
            loop_j(i, range(width))
            loop_j(i, range(width - 1, -1, -1))
        for j in range(width):
            loop_i(range(height), j)
            loop_i(range(height - 1, -1, -1), j)
        return ndimage.morphology.binary_closing(mask).astype(np.uint8)

    @classmethod
    def __normalize(cls,
                    image_selection,
                    j_list,
                    end_selection):
        """
        Normalizes intensity in the selected part of the B-scan.

        :param image_selection: selected part of the B-scan
        :param j_list: valid A-scans
        :param end_selection: end of the valid range of pixels in each column of image_selection
        """

        # per-A-scan intensity model
        a_list, b_list, j_list_selection = [], [], []
        for j in j_list:
            if end_selection[j] > 0:
                target = cls.TOP_INTENSITY - np.asarray(range(end_selection[j]))
                fit = np.polyfit(image_selection[:end_selection[j], j], target, 1)
                b, a = float(fit[0]), float(fit[1])
                a_list.append(a)
                b_list.append(b)
                j_list_selection.append(j)
        if len(j_list_selection) == 0:
            return

        # robust fitting function
        def robust_fit(param_list):
            model = linear_model.HuberRegressor()
            model.fit(np.expand_dims(j_list_selection, -1), param_list)
            return model.coef_[0], model.intercept_

        # per-B-scan intensity model
        b_a, a_a = robust_fit(a_list)
        b_b, a_b = robust_fit(b_list)

        # matching intensities
        for j in j_list:
            a, b = a_a + j * b_a, a_b + j * b_b
            image_selection[:end_selection[j], j] = \
                a + image_selection[:end_selection[j], j] * b

    def __post_process(self,
                       segmentations,
                       mask):
        """
        Post-processes segmentations.

        :param segmentations: input segmentation map(s)
        :param mask: mask of the field ov view

        :return: the post-processed segmentation maps
        """
        post_processed_segmentations = []
        for segmentation in to_list(segmentations):
            height, width = mask.shape
            segmentation = cv2.resize(segmentation, (width, height),
                                      interpolation = cv2.INTER_CUBIC)
            segmentation = (255. * (segmentation > self.__cutoff_segmentation)).astype(np.uint8)

            # selects the largest component
            _, labels, stats, _ = cv2.connectedComponentsWithStats(segmentation, connectivity = 4)
            sizes = stats[:, -1]
            label = 1 + np.argmax(sizes[1:])
            component = (labels == label).astype(np.uint8)

            # filling holes
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation = cv2.drawContours(component, contours, 0, 1, -1)

            # cropping the field of view
            post_processed_segmentations.append(segmentation * mask)
        return post_processed_segmentations

    def __top_retina(self,
                     segmentations,
                     j):
        """
        Returns the first element of an A-scan inside the retina.

        :param segmentations: post-processed segmentation maps
        :param j: index of the A-scan inside the B-scan

        :return: index of the first element of the A-scan inside the retina, or None
        """
        tops = []
        for segmentation in segmentations:
            indices = np.where(segmentation[:, j] == 1)[0]
            tops.append(indices[0] if indices.shape[0] > 0 else None)
        if None in tops:
            return None
        tops = sorted(tops)
        return None if (tops[-1] - tops[0] > self.__max_diff_tops) else tops[len(tops) // 2]

    def process_image(self,
                      bscan,
                      bscan_segmentations):
        """
        Processes one B-scan.

        Multiple retina segmentations can be passed so that segmentation errors can be spotted.

        :param bscan: input B-scan
        :param bscan_segmentations: segmentation(s) for the B-scan

        :return: the pre-processed B-scan
        """

        # resizing the image
        if self.__image_size is not None:
            bscan = cv2.resize(bscan, self.__image_size, interpolation = cv2.INTER_CUBIC)
        height, width = bscan.shape
        assert self.__max_height <= height, 'Argument max_height must be <= image height.'

        # computing the mask
        mask = PreProcessor.__compute_mask(bscan)
        end_selection = self.__max_height * np.ones([width], dtype = np.uint32)

        # post-processing segmentations
        bscan_segmentations = self.__post_process(bscan_segmentations, mask)

        # validity checking function
        def valid(j):
            if bscan_beginning - self.__min_distance_mask < 0 or \
                    bscan_beginning + self.__min_height >= height:
                return False
            invalid_indices = np.where(mask[bscan_beginning - self.__min_distance_mask:
                                            bscan_beginning + self.__min_height, j] == 0)[0]
            return invalid_indices.shape[0] == 0

        # selecting valid A-scans
        j_list, bscan_beginning_list = [], []
        for j in range(width):
            bscan_beginning = self.__top_retina(bscan_segmentations, j)
            if bscan_beginning is not None:
                if valid(j):
                    j_list.append(j)
                    bscan_beginning_list.append(bscan_beginning)
                else:
                    end_selection[j] = 0
            else:
                end_selection[j] = 0

        # finding contiguous A-scan intervals
        splits = []
        for idx in range(1, len(j_list)):
            if j_list[idx - 1] != j_list[idx] - 1:
                splits.append(idx)
        splits.append(len(j_list))

        # Gaussian filtering in each A-scan interval
        output_begin_list, prev_split = [], 0
        for split in splits:
            output_begin_list += ndimage.gaussian_filter1d(
                np.asarray(bscan_beginning_list[prev_split: split], dtype = np.float32),
                self.__sigma).tolist()
            prev_split = split
        bscan_beginning_list = output_begin_list

        # creating the output image
        image_selection = np.zeros([self.__max_height, width], dtype = np.float32)

        # clipping function
        def clip(value, max_value):
            return min(max_value, max(0, value))

        # warping the image selection
        for j, bscan_begin in zip(j_list, bscan_beginning_list):
            begin_floor = int(np.floor(bscan_begin))
            begin_inf = clip(begin_floor, height)
            begin_sup = clip(begin_floor + 1, height)
            factor_inf, factor_sup = begin_sup - bscan_begin, bscan_begin - begin_inf

            # sampling function
            def sampling(begin, factor, update_end_selection):
                length = clip(height - begin, self.__max_height)
                image_selection[:length, j] += \
                    factor * bscan[begin: begin + length, j]
                if update_end_selection:
                    invalid_indices = np.where(mask[begin: begin + length, j] == 0)[0]
                    if invalid_indices.shape[0] > 0:
                        end_selection[j] = min(end_selection[j], invalid_indices[0])
                    end_selection[j] = min(end_selection[j], length)

            # sampling
            sampling(begin_inf, factor_inf, True)
            sampling(begin_sup, factor_sup, False)

        # post-processing intensities
        if self.__normalize_intensities:
            PreProcessor.__normalize(image_selection, j_list, end_selection)
        for j in j_list:
            image_selection[end_selection[j]:, j] = 0
        return np.clip(image_selection, 0, 255).astype(np.uint8)
