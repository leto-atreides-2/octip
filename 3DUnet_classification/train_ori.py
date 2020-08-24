import os
import glob

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import unet_model_3d
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
#from unet3d.training import load_old_model, train_model
#在train.py中import 了unet3d文件夹中的函数

config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape1"] = (19, 200, 512)  # This determines what shape the images will be cropped/resampled to.
config["image_shape2"] = (61, 200, 768)
config["image_shape3"] = (19, 200, 1024)


config["patch_shape"] = (8, 128, 256)  # switch to None to train on the whole image
config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])

if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

#config["batch_size"] = 6
config["validation_batch_size"] = 12
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 0.00001
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("oct_data.h5")
config["model_file"] = os.path.abspath("tumor_segmentation_model.h5")
config["training_file"] = os.path.abspath("training_ids.pkl")
config["validation_file"] = os.path.abspath("validation_ids.pkl")


'''
def fetch_training_data_files():
    training_data_files = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed", "*", "*")):
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return training_data_files
#创建一个.nii.gz存放文件路径的列表
'''

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




def main(overwrite=False):
    training_files = fetch_training_data_files()
    print(training_files)
    write_data_to_file(training_files, config["data_file"], image_shape1=config["image_shape1"],image_shape2=config["image_shape2"], image_shape3=config["image_shape3"])

    data_file_opened = open_data_file(config["data_file"])
    print(data_file_opened)
    
    image_shape=(19, 200, 512)
    patch_shape=(8, 128, 256)
    batch_size=4

    #training_generator, validation_generator, num_training_steps, num_validation_steps = get_training_and_validation_generators(data_file_opened , batch_size, n_labels=1,data_split=0.8, patch_shape=patch_shape,validation_batch_size=batch_size, skip_blank=False, patch_overlap=0 )
    training_generator, validation_generator, test_generator, num_training_steps, num_validation_steps, num_test_steps = get_training_and_validation_generators(data_file_opened , batch_size, n_labels=1,data_split=0.8, patch_shape=patch_shape,validation_batch_size=batch_size, skip_blank=False, patch_overlap=0 )
    print("当前路径:")
    print(os.path.abspath('.'))
    os.chdir("/data_GPU/yihao/3dunet/final/original")
    print("当前路径1:")
    print(os.path.abspath('.'))
    

    model1 = unet_model_3d(input_shape=(1, 8, 128, 256), pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=3, n_base_filters=32,batch_normalization=False, activation_name="sigmoid")
    # 打印网络结构

    
    model1.summary()

    # 产生网络拓补图
    #plot_model(model1,to_file='convolutional_neural_network.png')
   
    MODEL_path = './model/'
    if not os.path.exists(MODEL_path):
        os.mkdir(MODEL_path)
    checkpoit = ModelCheckpoint(filepath=os.path.join(MODEL_path, 'model-{epoch:02d}.h5'))
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)          

    H=model1.fit_generator(generator=training_generator,
                        steps_per_epoch=num_training_steps,
                        validation_data=validation_generator,
                        validation_steps=num_validation_steps,
                        epochs=100,workers=0, callbacks=[checkpoit, tensorboard])

    data_file_opened.close()



if __name__ == "__main__":
    main()
