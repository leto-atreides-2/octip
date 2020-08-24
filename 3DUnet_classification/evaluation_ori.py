import numpy as np  #导入数据库
import tables
import os
from random import shuffle
import itertools
import copy
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def get_file_path(root_path,file_list,dir_list):
    #获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        #获取目录或者文件的路径
        dir_file_path = os.path.join(root_path,dir_file)
        #判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            #递归获取所有文件和目录的路径
            get_file_path(dir_file_path,file_list,dir_list)
        else:
            file_list.append(dir_file_path)

def filtre_ODOG(a,b,c):
    my_list = list()
    d = 'OD/'
    e = 'OG/'
    f = 'octip_data'
    for index,nums in enumerate(a):
      if not d in nums:
        if not e in nums:
          if not f in nums:
            if b in nums:
              my_list.append(nums)
            elif c in nums:
              my_list.append(nums)
    return my_list



def fetch_training_data_files():
    training_data_files = list()
    root_path = '/data_GPU/yihao/3dunet/final/OCT_Images'
    #用来存放所有的文件路径
    file_list = []
    #用来存放所有的目录路径 Utilisé pour stocker tous les chemins de répertoire
    dir_list = []
    get_file_path(root_path,file_list,dir_list)
    a = dir_list
    b = 'OD'
    c = 'OG'
    list_ODOG=filtre_ODOG(a,b,c)
    return list_ODOG


def compute_patch_indices(image_shape, patch_size, overlap=0, start=None):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
    if start is None:
        n_patches = np.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    return get_set_of_patch_indices(start, stop, step)

def create_patch_index_list(index_list, image_shape, patch_shape):
    patch_index = list()
    for index in index_list:
        patches = compute_patch_indices(image_shape, patch_shape)
        patch_index.extend(itertools.product([index], patches))
    return patch_index

def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)

def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

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

def get_patch_from_3d_data(data, patch_shape, patch_index):
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]

def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index

def convert_data(x_list, y_list):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    return x, y

def main():
  
  os.chdir("/data_GPU/yihao/3dunet/final")
  data_file = os.path.abspath("oct_data.h5")
  #print(os.path.abspath("oct_data.h5"))
  data_file_opened = open_data_file(data_file)
  print(data_file_opened)

  image_shape=(19, 200, 512)
  patch_shape=(8, 128, 256)
  batch_size=4
  test_list=[261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326]
  print('volume list pour base de testation:')
  print(test_list)
  print('nombre de testation:')
  print(len(test_list))

  test_index_list = create_patch_index_list(test_list, data_file_opened.root.data1.shape[-3:], patch_shape)
  print('patchs list dans base de testation et sa taille:')
  print(test_index_list)
  print(len(test_index_list))

  x_list = list()
  y_list = list()
  while len(test_index_list) > 0:
    index = test_index_list.pop()
    add_data(x_list, y_list, data_file_opened, index, patch_shape=patch_shape, skip_blank=False)
  x_arr,y_arr = convert_data(x_list, y_list)
  print('shape of x_arr and y_arr:')
  print(x_arr.shape)
  print(y_arr.shape)
  print('label=')
  print(y_arr)
  y_arr[y_arr==2]=1


  os.chdir("/data_GPU/yihao/3dunet/final")

  model_best=load_model('orimodel-17.h5')

  pred_list = model_best.predict(x_arr)
  #pred_list=pred_list.tolist()
  print('shape of prediction:')
  print(pred_list.shape)
  print('prediction=')
  print(pred_list)
  
  y_arr=y_arr.tolist()
  pred_list=pred_list.tolist()
  volume_label=[]
  volume_predic=[]
 
  for num in range(66):
    number=num*12
    label_current=max(y_arr[number:number+12])
    pred_current=max(pred_list[number:number+12])
    volume_label.append(label_current)
    volume_predic.append(pred_current)
  print('len volume_label/predic=')
  print(len(volume_label))
  print(len(volume_predic))
  print('pour volume,label et prediction=')
  print(volume_label)
  print(volume_predic)
  
  '''
  print('record the resultat in csv....')

  os.chdir("/data_GPU/yihao/3dunet/final")
  training_files = fetch_training_data_files()
  nombre_patch_dans_volume = 12
  nombre_volume_test=len(test_list)
  path_volume_test=training_files[-nombre_volume_test:]
  #print(path_volume_test)
  #print(len(path_volume_test))
  list_volume_test=[]
  #n=0
  for num in range(nombre_volume_test):
    path_current_volume= path_volume_test.pop()
    for numbre in range(nombre_patch_dans_volume):
      list_volume_test.append(path_current_volume)
  #print(list_volume_test)
  #print(len(list_volume_test))

  list_de_patchs=[]
  for a in range(nombre_volume_test):
    current_nombre=nombre_patch_dans_volume
    for b in range(nombre_patch_dans_volume):
      list_de_patchs.append(current_nombre)
      current_nombre=current_nombre-1
  #print(list_de_patchs)


  dfnew = pd.DataFrame(y_arr,columns=['label_of_volume/patchs'])
  dfnew['valeur_predict'] = pred_list
  dfnew.insert(0,'numbre of patchs in volume',list_de_patchs)
  dfnew.insert(0,'path_of_volume',list_volume_test)
  print(dfnew)
  dfnew.to_csv("resultat.csv")
  print('resultat has saved in csv with the path:')
  print(os.getcwd())
  '''
  print('dessiner la courbe ROC...')
  y_arr=volume_label
  pred_list=volume_predic

  fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_arr, pred_list)
  auc_keras = auc(fpr_keras,tpr_keras)
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.savefig('roc_ori.png')
  plt.show()
  print('resultat has saved in jpg with the path:')
  print(os.getcwd())

if __name__ == "__main__":
    main()
