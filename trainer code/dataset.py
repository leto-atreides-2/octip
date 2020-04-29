'''
Dataset for training
the input format of AHNET :[n , c , w, h ,d ]
'''

# 16-04-2020 without consideration of cross validation

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel
from scipy import ndimage


class AHNETdataset(Dataset):
    """
    Creat dataset for AHNET.
    - drop out the invalid range
    - crop data using random center crop
    - resize data with interpolation
    - volume transpose  from [d , h , w] to [w , h , d]
    """
    def __init__(self, root_dir, img_list, input_W, input_H, input_D, phase):
        """
        input parameters :
        root_dir : dataset path
        img_list : txt file path
        input_W  : expected width of volume
        input_H  : expected height of volume
        input_D  : expected depth of volume
        phase    : only "train" for now

        output :
        img_array : dtype float32 , [ input_W , input_H , input_D]
        mask_array : dtype float32 , [ input_W , input_H , input_]
        """

        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
            print("Processing {} datas".format(len(self.img_list)))
            self.root_dir = root_dir
            self.input_D = input_D
            self.input_H = input_H
            self.input_W = input_W
            self.phase = phase

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        if self.phase == "train":
            ith_info = self.img_list[idx].split(" ")
            img_name = os.path.join(ith_info[0])
            label_name = os.path.join(ith_info[1])
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            img = nibabel.load(img_name)
            assert img is not None
            mask = nibabel.load(label_name)
            assert mask is not None

            # data processing
            img_array, mask_array = self.__training_data_process__(img, mask)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            assert img_array.shape == mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(
                img_array.shape, mask_array.shape)
            return img_array, mask_array

    def __nii2tensorarray__(self, data):
        data = np.transpose(data, (2, 1, 0)) #transpose from [d , w , h] to [h , w ,d]
        [x, y, z] = data.shape
        new_data = np.reshape(data, [1, 1, x, y, z])
        new_data = new_data.astype("float32")
        return new_data

    def __training_data_process__(self, data, label):
        # crop data according net input size
        data = data.get_data()
        label = label.get_data()

        # drop out the invalid range
        data, label = self.__drop_invalid_range__(data, label)

        # crop data
        data, label = self.__crop_data__(data, label)

        # resize data
        data = self.__resize_data__(data)
        label = self.__resize_data__(label)

        return data, label

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D * 1.0 / depth, self.input_H * 1.0 / height, self.input_W * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data, label):
        """
        Random crop with different methods:
        """
        # random center crop
        data, label = self.__random_center_crop__(data, label)

        return data, label

    def __random_center_crop__(self, data, label):
        from random import random
        """
        Random crop
        """
        target_indexs = np.where(label > 0)
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth * 1.0 / 2) * random())
        Y_min = int((min_H - target_height * 1.0 / 2) * random())
        X_min = int((min_W - target_width * 1.0 / 2) * random())

        Z_max = int(img_d - ((img_d - (max_D + target_depth * 1.0 / 2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height * 1.0 / 2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width * 1.0 / 2)) * random()))

        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])

        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)

        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]


if __name__ == '__main__':
    num_workers = 0
    model_depth = 10
    input_D = 3
    input_H = 448
    input_W = 448
    phase = "train"

    training_dataset = AHNETdataset('./data', './data//txt_label.txt', input_W, input_H, input_D, phase)
    data_loader = DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    print('Finished')
