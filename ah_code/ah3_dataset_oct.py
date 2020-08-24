'''
Dataset for training
the input format of AHNET :[n , c , w, h ,d ]
'''

# 16-04-2020 without consideration of cross validation

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel
import io
from scipy import ndimage
import imageio
import torch


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    # input_tensor = input_tensor.clone().detach()
    # input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = img2d(input_tensor)
    imageio.imwrite(filename, input_tensor)


def img2d(input_tensor: torch.Tensor):
    if input_tensor.ndim == 3:
        input_tensor = input_tensor[0, :, :]
    elif input_tensor.ndim == 2:
        input_tensor = input_tensor
    elif input_tensor.ndim == 4:
        input_tensor = input_tensor[0, 0, :, :]
    elif input_tensor.ndim == 5:
        input_tensor = input_tensor[0, 0, 0, :, :]
    return input_tensor


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

    def __load_img__(self, path):

        img = imageio.imread(path + str(1) + '.png')
        [y, z] = img.shape
        volume_0 = np.zeros(((self.input_D * 2, y, z)))
        for idx in range(self.input_D):
            if str(idx) + '.png' in os.listdir(path):
                img = imageio.imread(path + str(idx) + '.png')

                volume_0[idx, :, :] = img
                depth_max = idx
        return volume_0[0:self.input_D , :, :]

    def __getitem__(self, idx):
        if True:
            ith_info = self.img_list[idx].split(" + ")
            img_file = os.path.join(ith_info[0])
            label_class = os.path.join(ith_info[1])
            label_class = float(label_class)
            img_array = self.__load_img__(img_file)
            assert img_array is not None
            assert label_class is not None
            # data processing

 #            if np.max(img_array)== np.min(img_array):
 #                img_array = (img_array - np.min(img_array))*250 - 125
 #            else :
 #                img_array = (img_array - np.min(img_array))*350/(np.max(img_array)-np.min(img_array))-125

            img_array, mask_array = self.__training_data_process__(img_array, label_class)
            img_array = img_array - img_array.mean()
            # 2 tensor array
            img_array = self.__img2tensorarray__(img_array)
            img_array = (img_array - img_array.mean())/ img_array.std()
            # for n in range(img_array.shape[2]) :
            #     save_image_tensor2cv2(img_array[0,0,n,:,:],  "./img_ah/ah_%s_img.png" % ( n))
            return img_array, mask_array

    def __img2tensorarray__(self, data):
        data = np.transpose(data, (2, 1, 0))
        if self.phase == '2d':
            [x, y, z] = data.shape
            new_data = np.reshape(data, [1, x, y, z])
        else :
            [x, y, z] = data.shape
            new_data = np.reshape(data, [1, 1, x, y, z])
        new_data = new_data
        new_data = new_data.astype("float32")
        return new_data

    def __training_data_process__(self, data, label):
        # crop data according net input size
        data = self.__drop_invalid_range__(data)
        data = self.__resize_data__(data )
        return data, label
    def __drop_invalid_range__(self, volume):
        """
        Cut off the i        # resize data
        # data = self.__resize_data__(data)
        # label = self.__resize_data__(label) nvalid area
        """
        zero_value = 0
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        return volume[:, min_h:max_h, min_w:max_w]

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]
        new_data = ndimage.interpolation.zoom(data, scale, order=0)
        return new_data

def my_collate(batch):
    nb_patch = 1
    for item in batch:
        if nb_patch < len(item[0]):
            [h, x, y, z] = item[0].shape
            patch_data = item[0].unfold(0, nb_patch, nb_patch)
            patch_label = item[1]
            patch_data = patch_data.permute(0, 4, 1, 2, 3).contiguous().view(-1, nb_patch, x, y, z)
            return [patch_data, patch_label]
        else:
            patch_data = item[0]
            patch_label = item[1]
            return [patch_data.unsqueeze(0), patch_label]
    return batch

if __name__ == '__main__':
    num_workers = 0
    model_depth = 10
    input_D = 19
    input_H = 200
    input_W = 1020
    phase = "train"

    training_dataset = AHNETdataset('./data', './oct_label.txt', input_W, input_H, input_D, phase)
    data_loader = DataLoader(training_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)

    print('Finished')
