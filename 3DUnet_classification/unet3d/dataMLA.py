import os

import numpy as np
import tables

from .normalize import normalize_data_storage, reslice_image_set


def create_data_file(out_file, n_channels, n_samples, image_shape1, image_shape2, image_shape3):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape1 = tuple([0, n_channels] + list(image_shape1))
    data_shape2 = tuple([0, n_channels] + list(image_shape2))
    data_shape3 = tuple([0, n_channels] + list(image_shape3))
    #truth_shape = tuple([0, 1] + list(image_shape))
    truth_shape = tuple([0, 1, 1])
    data_storage1 = hdf5_file.create_earray(hdf5_file.root, 'data1', tables.Float32Atom(), shape=data_shape1,
                                           filters=filters)
    data_storage2 = hdf5_file.create_earray(hdf5_file.root, 'data2', tables.Float32Atom(), shape=data_shape2,
                                           filters=filters)
    data_storage3 = hdf5_file.create_earray(hdf5_file.root, 'data3', tables.Float32Atom(), shape=data_shape3,
                                           filters=filters)
    truth_storage1 = hdf5_file.create_earray(hdf5_file.root, 'truth1', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters)
    truth_storage2 = hdf5_file.create_earray(hdf5_file.root, 'truth2', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters)
    truth_storage3 = hdf5_file.create_earray(hdf5_file.root, 'truth3', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters)
    return hdf5_file, data_storage1, data_storage2, data_storage3, truth_storage1, truth_storage2, truth_storage3



#函数write_image_data_to_file()的作用是向之前创建的压缩可扩展的数组中写入图像数据．
def write_image_data_to_file(training_files, data_storage1, data_storage2, data_storage3, truth_storage1, truth_storage2, truth_storage3):
    for index,nums in enumerate(training_files):
      path_current = nums
      os.chdir(path_current)
      cscan=np.load('cscan_array.npy')
      #print(image.shape)
      label=np.load('label_MLA.npy')
      #print(label.shape)
      image_shape1 = (19, 200, 512)
      if cscan.shape[0]>= 19:
        over_z=cscan.shape[0]-image_shape1[0]
        over_x=cscan.shape[1]-image_shape1[1]
        over_y=cscan.shape[2]-image_shape1[2]
        cscan_new=cscan[over_z//2:over_z//2+image_shape1[0],over_x//2:over_x//2+image_shape1[1],over_y//2:over_y//2+image_shape1[2]]
        add_data_to_storage(data_storage1, truth_storage1, cscan_new, label)
    return data_storage1, data_storage2, data_storage3, truth_storage1, truth_storage2, truth_storage3


#首先是获取一组图像的路径，然后读取图像数据，再利用函数add_data_to_storage()，将获取到的图像数据写入压缩可扩展数组．
def add_data_to_storage(data_storage, truth_storage, data, label):
    data_storage.append(data[np.newaxis][np.newaxis])
    truth_storage.append(label[np.newaxis][np.newaxis][np.newaxis])





def write_data_to_file(training_data_files, out_file, image_shape1, image_shape2, image_shape3, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True):
    """
    Takes in a set of training images and writes those images to an hdf5 file.
    :param training_data_files: List of tuples containing the training data files. The modalities should be listed in
    the same order in each tuple. The last item in each tuple must be the labeled image.
    Example: [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'),
              ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
    :param out_file: Where the hdf5 file will be written to.
    :param image_shape: Shape of the images that will be saved to the hdf5 file.
    :param truth_dtype: Default is 8-bit unsigned integer.
    :return: Location of the hdf5 file with the image data written to it.
    """
    n_samples = len(training_data_files)
    #n_channels = len(training_data_files[0]) - 1
    n_channels=1

    try:
      hdf5_file, data_storage1, data_storage2, data_storage3, truth_storage1, truth_storage2, truth_storage3 = create_data_file(out_file, n_channels, n_samples, image_shape1, image_shape2, image_shape3)
    except Exception as e:
        os.remove(out_file)
        raise e


    write_image_data_to_file(training_data_files, data_storage1, data_storage2, data_storage3, truth_storage1, truth_storage2, truth_storage3)

    hdf5_file.close()
    return out_file



#最后是函数open_data_file()是读取table文件的数据．
def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
