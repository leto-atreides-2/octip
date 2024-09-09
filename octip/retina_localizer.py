"""
OCTIP retina localization module.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'GPL-v3'
__version__ = '1.1'

import cv2
import numpy as np
import os
import segmentation_models as sm
import tensorflow as tf
from .utils import extract_base_name


class RetinaLocalizationDataset(tf.keras.utils.Sequence):
    """
    B-scan dataset prepared for retina localization (for prediction only, not suitable for training).
    """

    def __init__(self,
                 samples,
                 batch_size,
                 retina_localizer):
        """
        DataGenerator constructor.

        :param samples: list of images to process (absolute or relative path to image files)
        :param batch_size: batch size
        :param retina_localizer: RetinaLocalizer instance
        """
        self.__samples = samples
        self.__num_samples = len(samples)
        self.__batch_size = batch_size
        self.__indices = np.arange(self.__num_samples)
        self.__pre_processing_function = retina_localizer.pre_processing_function()
        self.__image_size = retina_localizer.image_size

    def __getitem__(self,
                    index):
        """
        Generates one batch of data.

        :param index: batch index

        :return: a batch a data
        """
        indices = self.__sample_indices(index)
        x = np.asarray([cv2.cvtColor(self.__load_resize_image(self.__samples[index]),
                                     cv2.COLOR_GRAY2RGB) for index in indices])
        return self.__pre_processing_function(x.astype(float))

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        :return: number of batches per epoch
        """
        return int(np.ceil(self.__num_samples / self.__batch_size))

    def __load_resize_image(self,
                            input_file_name):
        """
        Loads and pre-processes an image.

        :param input_file_name: input image file name

        :return: the pre-processed image
        """
        image = cv2.imread(input_file_name, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(image, self.__image_size, interpolation = cv2.INTER_CUBIC)

    def __sample_indices(self,
                         index):
        """
        Converts a batch index to sample indices.

        :param index: batch index

        :return: the index of images in the batch
        """
        return self.__indices[index * self.__batch_size: (index + 1) * self.__batch_size]

    def file_names(self):
        """
        List of images to process (absolute or relative path to image files).

        :return: the list of images to process
        """
        return self.__samples


class RetinaLocalizer(object):
    """
    Localizes the retina B-scans.
    """

    def __init__(self,
                 architecture,
                 encoder,
                 image_size,
                 model_directory = ''):
        """
        RetinaLocalizer constructor.

        :param architecture: name of the architecture (one of 'Unet', 'Linknet', 'PSPNet' and 'FPN')
        :param encoder: name of the backbone CNN used as encoder
        :param image_size: (width, height) of input images
        :param optimizer: optimization object
        :param model_directory: directory containing the models
        """

        # Keras and TensorFlow initialization
        tf.keras.backend.set_image_data_format('channels_last')
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # class member initialization
        self.__architecture = architecture
        self.__encoder = encoder
        self.image_size = image_size
        self.__metrics = [sm.metrics.iou_score, sm.metrics.f1_score]
        self.__model_file = os.path.join(
            model_directory,
            '{}_{}_{}x{}.hdf5'.format(architecture, encoder, image_size[0], image_size[1]))
        assert os.path.exists(self.__model_file), \
            'Model file \'{}\' does not exist!'.format(self.__model_file)
        self.__create_model()

    def __call__(self,
                 dataset,
                 output_directory,
                 num_workers = 1):
        """
        Segments B-scans and saves segmentations.

        :param dataset: RetinaLocalizationDataset instance
        :param output_directory: output image directory
        :param num_workers: number of processes
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_names, segmentations = self.process_image(dataset, num_workers)
        for file_name, segmentation in zip(file_names, segmentations):
            base_name = extract_base_name(file_name)
            output_file = os.path.join(output_directory, base_name + '.png')
            cv2.imwrite(output_file, (255 * segmentation).astype(np.uint8))

    def __architecture_cls(self):
        """
        Gets the architecture class from the architecture name.

        :return: the architecture class
        """
        if self.__architecture == 'Unet':
            return sm.Unet
        elif self.__architecture == 'Linknet':
            return sm.Linknet
        elif self.__architecture == 'PSPNet':
            return sm.PSPNet
        elif self.__architecture == 'FPN':
            return sm.FPN
        else:
            raise ValueError('Architecture should be one of Unet, Linknet, PSPNet and FPN')

    def __create_model(self):
        """
        Creates the model.
        """
        self.__model = self.__architecture_cls() \
            (self.__encoder, classes = 1, activation = 'sigmoid',
             input_shape = (self.image_size[1], self.image_size[0], 3),
             encoder_weights = 'imagenet')
        self.__model.load_weights(self.__model_file)
        self.__model.compile(loss = sm.losses.binary_focal_jaccard_loss, metrics = self.__metrics)

    def pre_processing_function(self):
        """
        Encoder-specific batch pre-processing function to use with this model.

        :return: the batch pre-processing function
        """
        return sm.get_preprocessing(self.__encoder)

    def process_image(self,
                      dataset,
                      num_workers = 1):
        """
        Segments B-scans.

        :param dataset: RetinaLocalizationDataset instance
        :param num_workers: number of processes

        :return: the list of images to process (absolute or relative path to image files)
        :return: the segmentation for each image
        """
        return dataset.file_names(), \
            self.__model.predict(dataset, verbose = 1,
                                 use_multiprocessing = (num_workers > 1), workers = num_workers)
