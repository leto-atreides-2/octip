import os
import copy
from random import shuffle
import itertools

import numpy as np

from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y


def get_training_and_validation_generators(data_file, batch_size, n_labels=1,data_split=0.8, patch_shape=None,validation_batch_size=None, skip_blank=True, patch_overlap=0 ):
    
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list, test_list = get_validation_split(data_file,data_split=0.8)
    print(training_list)
    print(validation_list)
    print(test_list)

    training_generator= data_generator(data_file, training_list,
                          batch_size=batch_size,
                          n_labels=1,
                          patch_shape=patch_shape,
                          patch_overlap=0,
                          skip_blank=False)
    validation_generator = data_generator(data_file, validation_list,
                          batch_size= validation_batch_size,
                          n_labels=1,
                          patch_shape=patch_shape,
                          patch_overlap=0,
                          skip_blank=False)
    test_generator = data_generator(data_file, test_list,
                          batch_size= 1,
                          n_labels=1,
                          patch_shape=patch_shape,
                          patch_overlap=0,
                          skip_blank=False)
    #print("succ")
    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape=None,skip_blank=False), batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape=None,skip_blank=False), batch_size)
    print("Number of validation steps: ", num_validation_steps)

    num_test_steps = get_number_of_steps(get_number_of_patches(data_file, test_list, patch_shape=None,skip_blank=False), batch_size=1)
    print("Number of test steps: ", num_test_steps)

    return training_generator, validation_generator, test_generator, num_training_steps, num_validation_steps, num_test_steps


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_validation_split(data_file,data_split=0.8):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    print("Creating training, validation, test split...")
    nb_samples = data_file.root.data1.shape[0]
    train_nombre = int(0.8 * nb_samples)
    sample_list = list(range(nb_samples))
    train_list = sample_list[:train_nombre]
    test_list = sample_list[train_nombre:]
    training_list, validation_list = split_list(train_list, split=data_split)
    return training_list, validation_list, test_list 


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing


def data_generator(data_file, index_list, batch_size=1, n_labels=1, patch_shape=None, patch_overlap=0, shuffle_index_list=True, skip_blank=False):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data1.shape[-3:], patch_shape)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, patch_shape=patch_shape, skip_blank=skip_blank)
            #if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
            if len(x_list) == batch_size:
                yield convert_data(x_list, y_list)
                x_list = list()
                y_list = list()


def get_number_of_patches(data_file, index_list, patch_shape=None,skip_blank=False):
    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file.root.data1.shape[-3:], patch_shape)
        count = 0
        for index in index_list:
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, patch_shape=False, skip_blank=False)
            if len(x_list) > 0:
                count += 1
        return count
    else:
        return len(index_list)


def create_patch_index_list(index_list, image_shape, patch_shape):
    patch_index = list()
    for index in index_list:
        patches = compute_patch_indices(image_shape, patch_shape)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def add_data(x_list, y_list, data_file, index, patch_shape=False, skip_blank=False):
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = data_file.root.truth1[index]
        y = np.array(y)
        y = y.reshape(1,)
    else:
        x, y = data_file.root.data1[index], data_file.root.truth1[index]
    return x, y


def convert_data(x_list, y_list):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y
