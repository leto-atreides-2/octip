[Keras] 3D U-Net源码解析之generator.py
================

本文主要介绍在generator.py中提到的函数
主要就是主函数调用的get_training_and_validation_generators

# get_training_and_validation_generators
这个函数主要是为了我们之后使用fit_generator做准备，产生可在训练模型时使用的训练集和验证集生成器。
```
def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite=False, labels=None, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                                           validation_patch_overlap=0, training_patch_start_offset=None,
                                           validation_batch_size=None, skip_blank=True, permute=False):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_file=training_keys_file,
                                                          validation_file=validation_keys_file)

    training_generator = data_generator(data_file, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        permute=permute)
    validation_generator = data_generator(data_file, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank)

    # Set the number of training and testing samples per epoch correctly
    num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape,
                                                                   skip_blank=skip_blank,
                                                                   patch_start_offset=training_patch_start_offset,
                                                                   patch_overlap=0), batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape,
                                                                     skip_blank=skip_blank,
                                                                     patch_overlap=validation_patch_overlap),
                                               validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps
```
## 参数
我将在train.py文件中调用时使用的所有的参数都写在了对应等号后边，方便大家查看。
1. skip_blank = config["skip_blank"] = True  # if True, then patches without any target will be skipped  
如果为True，则数据生成器将跳过任何空白（全零）标签图像/补丁。

2. validation_batch_size = config["validation_batch_size"] = 12  
验证数据的批量大小。

3. training_patch_start_offset =config["training_patch_start_offset"] = = (16, 16, 16)  # randomly offset the first patch index by up to this offset  
长度为3的元组，包含整数值。训练数据将随机偏移（0，0，0）与给定元组之间的许多像素。 （默认为无）

4. validation_patch_overlap = config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping  
在验证数据中将重叠的像素/体素数。 （要求patch_shape不为None）

5. patch_shape = config["patch_shape"] = (64, 64, 64)  # switch to None to train on the whole image   
要与生成器一起返回的数据的形状。如果为None，则将返回整个图像。（默认为无）
Patch是图像块的大小，比如说原图1024*1024，随机从图中裁剪出256*256大小的块，就是patch。

6. augment_flip = config["flip"] = False  # augments the data by randomly flipping an axis during   
如果为True，且augment为True，则数据将沿x，y和z轴随机翻转

7. augment_distortion_factor = config["distort"] = None  # switch to None if you want no distortion  
如果augment为True，则这将确定与原始数据的标准偏差，即数据将失真（以拉伸或收缩的方式）。如果设置为None，False或0，则不会使数据失真。

8. augment =config["augment"] = config["flip"] or config["distort"] = False  
如果为True，则训练数据将实时失真，以避免过度拟合。

9. labels = config["labels"] = (1, 2, 4)  # the label numbers on the input image  
列表或元组，其中包含图像文件中的标签顺序。列表或元组的长度应等于n_labels的值。
Example: (10, 25, 50)
数据生成器将返回表示标签10、25和30的二进制真值数组。    
如BraTS参考文件所述，label包括GD增强型肿瘤（ET-标签4），肿瘤周围水肿（ED-标签2）以及坏死和非增强型肿瘤核心（NCR / NET-标签1）
所以这里是1，2，4

10. data_file = data_file_opened    
从中加载数据的hdf5文件。之前用data.py中的函数产生的hdf5文件

11. batch_size = config["batch_size"] = 6  
训练生成器将提供的批次大小。
Batch就是每次送入网络中训练的一部分数据，而Batch Size就是每个batch中训练样本的数量  
假设您有一个包含200个样本（数据行）的数据集，并且您选择的Batch大小为5和1,000个Epoch。  
这意味着数据集将分为40个Batch，每个Batch有5个样本。每批五个样品后，模型权重将更新。  
这也意味着一个epoch将涉及40个Batch或40个模型更新。  
有1000个Epoch，模型将暴露或传递整个数据集1,000次。在整个培训过程中，总共有40,000Batch。

12. n_labels = len(config["labels"]) =  3  
二进制标签数。

13.  training_keys_file = config["training_file"] = os.path.abspath("training_ids.pkl")
Pickle文件，将在其中存储训练数据的索引位置。

14. validation_keys_file = config["validation_file"] = os.path.abspath("validation_ids.pkl")   
将存储验证数据的索引位置的Pickle文件。

15. data_split = config["validation_split"] = 0.8  
如何分割训练和验证数据。 0表示所有数据都将用于验证，而所有数据都不会用于训练。 1表示所有数据都将用于训练，而没有数据将用于验证。 默认值为0.8或80％。

16. overwrite = True  # If True, will previous files. If False, will use previously written files.  
如果设置为True，则先前的文件将被覆盖。 默认模式为false，因此重新运行模型训练时，训练和验证拆分不会被覆盖。

17. permute =c onfig["permute"] =  True  # data shape must be a cube. Augments the data by permuting in various directions    
将随机置换数据（数据必须是3D立方体）

18. 返回
训练数据生成器，验证数据生成器，训练步骤数，验证步骤数

## 第一块代码
```
if not validation_batch_size:
    validation_batch_size = batch_size
```
如果不定义验证集的batch大小，则将其设置为与训练集的batch一样大

## 第二块代码
```
training_list, validation_list = get_validation_split(data_file,
                                                      data_split=data_split,
                                                      overwrite=overwrite,
                                                      training_file=training_keys_file,
                                                      validation_file=validation_keys_file)

```
将数据分为训练和验证索引列表,函数详解见下文。

## 第三块代码
```
training_generator = data_generator(data_file, training_list,
                                    batch_size=batch_size,
                                    n_labels=n_labels,
                                    labels=labels,
                                    augment=augment,
                                    augment_flip=augment_flip,
                                    augment_distortion_factor=augment_distortion_factor,
                                    patch_shape=patch_shape,
                                    patch_overlap=0,
                                    patch_start_offset=training_patch_start_offset,
                                    skip_blank=skip_blank,
                                    permute=permute)
validation_generator = data_generator(data_file, validation_list,
                                      batch_size=validation_batch_size,
                                      n_labels=n_labels,
                                      labels=labels,
                                      patch_shape=patch_shape,
                                      patch_overlap=validation_patch_overlap,
                                      skip_blank=skip_blank)
```
这一块其实就是产生训练集及验证集的数据生成器，以便以后在fit_generator中使用到。

## 第四块代码
```
# Set the number of training and testing samples per epoch correctly
num_training_steps = get_number_of_steps(get_number_of_patches(data_file, training_list, patch_shape,
                                                               skip_blank=skip_blank,
                                                               patch_start_offset=training_patch_start_offset,
                                                               patch_overlap=0), batch_size)
print("Number of training steps: ", num_training_steps)

num_validation_steps = get_number_of_steps(get_number_of_patches(data_file, validation_list, patch_shape,
                                                                 skip_blank=skip_blank,
                                                                 patch_overlap=validation_patch_overlap),
                                           validation_batch_size)
print("Number of validation steps: ", num_validation_steps)
```
通过这块代码计算出num_training_steps与num_validation_steps，以便以后在fit_generator中使用到。

## 第五块代码
```
return training_generator, validation_generator, num_training_steps, num_validation_steps
```
最终，我们的函数返回这四个参数，供下面训练时fit_generator函数调用  
写到这花了十几个小时，神志不清了，一个好的项目果然考虑的东西足够多呀~  

# get_validation_split
定义为
```
def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.data.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)
```
将数据分为训练和验证索引列表。  
我们一行行分析一下,  
当overwrite=True的时候，  
通过我们写入完成的可压缩数组获取总样本数的大小。详细请看我上一篇data.py中的create_data_file函数，在创建函数时，储存训练图像数据的可压缩数组data_storage，这里的shape[0]是因为可压缩数组的每一组数据都在第一维叠加。  
sample_list是产生的总图像数据的索引列表。
通过split_list来划分为训练集索引列表与验证集索引列表，函数见下文。    
## split_list
```
def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing
```
这个函数其实很简单，先洗牌，取split的值也就是前百分之八十作为训练集索引列表，后百分之二十作为验证集索引列表，然后返回。  

再得到训练集索引列表与验证集索引列表之后，用pickle_dump将他们保存到文件中。
```
def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)
```
Python中的Pickle模块实现了基本的数据序列与反序列化。  
dump()方法

pickle.dump(obj, file, [,protocol])

注释：序列化对象，将对象obj保存到文件file中去。参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）。file表示保存到的类文件对象，file必须有write()接口，file可以是一个以'w'打开的文件或者是一个StringIO对象，也可以是任何可以实现write()接口的对象。

# data_generator
```
def data_generator(data_file, index_list, batch_size=1, n_labels=1, labels=None, augment=False, augment_flip=True,
                   augment_distortion_factor=0.25, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                   shuffle_index_list=True, skip_blank=True, permute=False):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                                 patch_overlap, patch_start_offset)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor, patch_shape=patch_shape,
                     skip_blank=skip_blank, permute=permute)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()
```
这是我们数据生成器的核心函数，比较麻烦，让我们一点点分析。  
index_list 这里输入不同的索引列表产生训练或验证集数据生成器  
while True: 数据生成器函数的标准格式，不是很熟悉的同学可以看我之前fit_generator文章中有讲  
x_list = list()用来记录训练/验证集图像数据
y_list = list()用来纪录标签图像数据
```
if patch_shape:
    index_list = create_patch_index_list(orig_index_list, data_file.root.data.shape[-3:], patch_shape,
                                         patch_overlap, patch_start_offset)
else:
    index_list = copy.copy(orig_index_list)
```  
如果patch_shape=None，则会返回整张图象，但基本很少这么做，因为如果patch_shape很大，会在训练时超出内存限制。    
现在让我们分析一下create_patch_index_list，见下文  
我们通过create_patch_index_list创建patch索引列表  
```
if shuffle_index_list:
    shuffle(index_list)
while len(index_list) > 0:
    index = index_list.pop()
    add_data(x_list, y_list, data_file, index, augment=augment, augment_flip=augment_flip,
             augment_distortion_factor=augment_distortion_factor, patch_shape=patch_shape,
             skip_blank=skip_blank, permute=permute)
```
如果洗牌的话，就打乱一下我们的index_list  
这里为了便于观察输出，就不打乱了  
然后执行index = index_list.pop()，pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。
因为while len(index_list) > 0，所以我们的列表长度会不停的减少直至为零。  
也就是说，我们从之前27*24个索引中一个patch一个patch的读，    
我们一共有24组训练数据，所以从最后一组[23]的最后一个batch起始点[104 104 104]开始读入patch数据，执行add_data来产生patch的数据和标签，执行完后再执行最后一组[23]的倒数第二个batch起始点。。。以此类推，知道读完27*24个patch的数据        
接下来开始分析我们的add_data函数，他将数据文件中的数据添加到要素和目标数据的给定列表中,详见下文
```
if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
    yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
    x_list = list()
    y_list = list()
```
最后在用add_data函数得到所有的patch数据后，用if语句判断是否执行完毕  
若遍历完毕，则执行convert_data函数，通过我们确定的标签个数最后对数据进行微调
最后通过yield返回完patch数据及标签后  
把x_list及y_list清零，重新进入循环来产生新的数据   

## create_patch_index_list
```
def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index
```
这里我们分析patch_start_offset=None，patch_overlap=0的情况，  
for index in index_list对索引列表进行循环，每个索引执行compute_patch_indices函数  
compute_patch_indices函数解析见下文  
```
patch_index.extend(itertools.product([index], patches))  
```
extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

itertools.product(iterables, repeat=1)  
可迭代对象输入的笛卡儿积。    
大致相当于生成器表达式中的嵌套循环。例如， product(A, B) 和 ((x,y) for x in A for y in B) 返回结果一样。   
嵌套循环像里程表那样循环变动，每次迭代时将最右侧的元素向后迭代。这种模式形成了一种字典序，因此如果输入的可迭代对象是已排序的，笛卡尔积元组依次序发出。    
要计算可迭代对象自身的笛卡尔积，将可选参数 repeat 设定为要重复的次数。例如，product(A, repeat=4) 和 product(A, A, A, A) 是一样的  

```
sample_list = list(range(30))
training = sample_list[:24]
testing = sample_list[24:]

import itertools
patch_index = list()
for index in training:
  patch_index.extend(itertools.product([index], patches))
print(patch_index)
print(len(patch_index))#27*24
# 输出
[(0, array([-24, -24, -24])), (0, array([-24, -24,  40])), (0, array([-24, -24, 104])), (0, array([-24,  40, -24])), (0, array([-24,  40,  40])), (0, array([-24,  40, 104])), (0, array([-24, 104, -24])), (0, array([-24, 104,  40])), (0, array([-24, 104, 104])), (0, array([ 40, -24, -24])), (0, array([ 40, -24,  40])), (0, array([ 40, -24, 104])), (0, array([ 40,  40, -24])), (0, array([40, 40, 40])), (0, array([ 40,  40, 104])), (0, array([ 40, 104, -24])), (0, array([ 40, 104,  40])), (0, array([ 40, 104, 104])), (0, array([104, -24, -24])), (0, array([104, -24,  40])), (0, array([104, -24, 104])), (0, array([104,  40, -24])), (0, array([104,  40,  40])), (0, array([104,  40, 104])), (0, array([104, 104, -24])), (0, array([104, 104,  40])), (0, array([104, 104, 104])), (1, array([-24, -24, -24])), (1, array([-24, -24,  40])), (1, array([-24, -24, 104])), (1, array([-24,  40, -24])), (1, array([-24,  40,  40])), (1, array([-24,  40, 104])), (1, array([-24, 104, -24])), (1, array([-24, 104,  40])), (1, array([-24, 104, 104])), (1, array([ 40, -24, -24])), (1, array([ 40, -24,  40])), (1, array([ 40, -24, 104])), (1, array([ 40,  40, -24])), (1, array([40, 40, 40])), (1, array([ 40,  40, 104])), (1, array([ 40, 104, -24])), (1, array([ 40, 104,  40])), (1, array([ 40, 104, 104])), (1, array([104, -24, -24])), (1, array([104, -24,  40])), (1, array([104, -24, 104])), (1, array([104,  40, -24])), (1, array([104,  40,  40])), (1, array([104,  40, 104])), (1, array([104, 104, -24])), (1, array([104, 104,  40])), (1, array([104, 104, 104])), (2, array([-24, -24, -24])), (2, array([-24, -24,  40])), (2, array([-24, -24, 104])), (2, array([-24,  40, -24])), (2, array([-24,  40,  40])), (2, array([-24,  40, 104])), (2, array([-24, 104, -24])), (2, array([-24, 104,  40])), (2, array([-24, 104, 104])), (2, array([ 40, -24, -24])), (2, array([ 40, -24,  40])), (2, array([ 40, -24, 104])), (2, array([ 40,  40, -24])), (2, array([40, 40, 40])), (2, array([ 40,  40, 104])), (2, array([ 40, 104, -24])), (2, array([ 40, 104,  40])), (2, array([ 40, 104, 104])), (2, array([104, -24, -24])), (2, array([104, -24,  40])), (2, array([104, -24, 104])), (2, array([104,  40, -24])), (2, array([104,  40,  40])), (2, array([104,  40, 104])), (2, array([104, 104, -24])), (2, array([104, 104,  40])), (2, array([104, 104, 104])), (3, array([-24, -24, -24])), (3, array([-24, -24,  40])), (3, array([-24, -24, 104])), (3, array([-24,  40, -24])), (3, array([-24,  40,  40])), (3, array([-24,  40, 104])), (3, array([-24, 104, -24])), (3, array([-24, 104,  40])), (3, array([-24, 104, 104])), (3, array([ 40, -24, -24])), (3, array([ 40, -24,  40])), (3, array([ 40, -24, 104])), (3, array([ 40,  40, -24])), (3, array([40, 40, 40])), (3, array([ 40,  40, 104])), (3, array([ 40, 104, -24])), (3, array([ 40, 104,  40])), (3, array([ 40, 104, 104])), (3, array([104, -24, -24])), (3, array([104, -24,  40])), (3, array([104, -24, 104])), (3, array([104,  40, -24])), (3, array([104,  40,  40])), (3, array([104,  40, 104])), (3, array([104, 104, -24])), (3, array([104, 104,  40])), (3, array([104, 104, 104])), (4, array([-24, -24, -24])), (4, array([-24, -24,  40])), (4, array([-24, -24, 104])), (4, array([-24,  40, -24])), (4, array([-24,  40,  40])), (4, array([-24,  40, 104])), (4, array([-24, 104, -24])), (4, array([-24, 104,  40])), (4, array([-24, 104, 104])), (4, array([ 40, -24, -24])), (4, array([ 40, -24,  40])), (4, array([ 40, -24, 104])), (4, array([ 40,  40, -24])), (4, array([40, 40, 40])), (4, array([ 40,  40, 104])), (4, array([ 40, 104, -24])), (4, array([ 40, 104,  40])), (4, array([ 40, 104, 104])), (4, array([104, -24, -24])), (4, array([104, -24,  40])), (4, array([104, -24, 104])), (4, array([104,  40, -24])), (4, array([104,  40,  40])), (4, array([104,  40, 104])), (4, array([104, 104, -24])), (4, array([104, 104,  40])), (4, array([104, 104, 104])), (5, array([-24, -24, -24])), (5, array([-24, -24,  40])), (5, array([-24, -24, 104])), (5, array([-24,  40, -24])), (5, array([-24,  40,  40])), (5, array([-24,  40, 104])), (5, array([-24, 104, -24])), (5, array([-24, 104,  40])), (5, array([-24, 104, 104])), (5, array([ 40, -24, -24])), (5, array([ 40, -24,  40])), (5, array([ 40, -24, 104])), (5, array([ 40,  40, -24])), (5, array([40, 40, 40])), (5, array([ 40,  40, 104])), (5, array([ 40, 104, -24])), (5, array([ 40, 104,  40])), (5, array([ 40, 104, 104])), (5, array([104, -24, -24])), (5, array([104, -24,  40])), (5, array([104, -24, 104])), (5, array([104,  40, -24])), (5, array([104,  40,  40])), (5, array([104,  40, 104])), (5, array([104, 104, -24])), (5, array([104, 104,  40])), (5, array([104, 104, 104])), (6, array([-24, -24, -24])), (6, array([-24, -24,  40])), (6, array([-24, -24, 104])), (6, array([-24,  40, -24])), (6, array([-24,  40,  40])), (6, array([-24,  40, 104])), (6, array([-24, 104, -24])), (6, array([-24, 104,  40])), (6, array([-24, 104, 104])), (6, array([ 40, -24, -24])), (6, array([ 40, -24,  40])), (6, array([ 40, -24, 104])), (6, array([ 40,  40, -24])), (6, array([40, 40, 40])), (6, array([ 40,  40, 104])), (6, array([ 40, 104, -24])), (6, array([ 40, 104,  40])), (6, array([ 40, 104, 104])), (6, array([104, -24, -24])), (6, array([104, -24,  40])), (6, array([104, -24, 104])), (6, array([104,  40, -24])), (6, array([104,  40,  40])), (6, array([104,  40, 104])), (6, array([104, 104, -24])), (6, array([104, 104,  40])), (6, array([104, 104, 104])), (7, array([-24, -24, -24])), (7, array([-24, -24,  40])), (7, array([-24, -24, 104])), (7, array([-24,  40, -24])), (7, array([-24,  40,  40])), (7, array([-24,  40, 104])), (7, array([-24, 104, -24])), (7, array([-24, 104,  40])), (7, array([-24, 104, 104])), (7, array([ 40, -24, -24])), (7, array([ 40, -24,  40])), (7, array([ 40, -24, 104])), (7, array([ 40,  40, -24])), (7, array([40, 40, 40])), (7, array([ 40,  40, 104])), (7, array([ 40, 104, -24])), (7, array([ 40, 104,  40])), (7, array([ 40, 104, 104])), (7, array([104, -24, -24])), (7, array([104, -24,  40])), (7, array([104, -24, 104])), (7, array([104,  40, -24])), (7, array([104,  40,  40])), (7, array([104,  40, 104])), (7, array([104, 104, -24])), (7, array([104, 104,  40])), (7, array([104, 104, 104])), (8, array([-24, -24, -24])), (8, array([-24, -24,  40])), (8, array([-24, -24, 104])), (8, array([-24,  40, -24])), (8, array([-24,  40,  40])), (8, array([-24,  40, 104])), (8, array([-24, 104, -24])), (8, array([-24, 104,  40])), (8, array([-24, 104, 104])), (8, array([ 40, -24, -24])), (8, array([ 40, -24,  40])), (8, array([ 40, -24, 104])), (8, array([ 40,  40, -24])), (8, array([40, 40, 40])), (8, array([ 40,  40, 104])), (8, array([ 40, 104, -24])), (8, array([ 40, 104,  40])), (8, array([ 40, 104, 104])), (8, array([104, -24, -24])), (8, array([104, -24,  40])), (8, array([104, -24, 104])), (8, array([104,  40, -24])), (8, array([104,  40,  40])), (8, array([104,  40, 104])), (8, array([104, 104, -24])), (8, array([104, 104,  40])), (8, array([104, 104, 104])), (9, array([-24, -24, -24])), (9, array([-24, -24,  40])), (9, array([-24, -24, 104])), (9, array([-24,  40, -24])), (9, array([-24,  40,  40])), (9, array([-24,  40, 104])), (9, array([-24, 104, -24])), (9, array([-24, 104,  40])), (9, array([-24, 104, 104])), (9, array([ 40, -24, -24])), (9, array([ 40, -24,  40])), (9, array([ 40, -24, 104])), (9, array([ 40,  40, -24])), (9, array([40, 40, 40])), (9, array([ 40,  40, 104])), (9, array([ 40, 104, -24])), (9, array([ 40, 104,  40])), (9, array([ 40, 104, 104])), (9, array([104, -24, -24])), (9, array([104, -24,  40])), (9, array([104, -24, 104])), (9, array([104,  40, -24])), (9, array([104,  40,  40])), (9, array([104,  40, 104])), (9, array([104, 104, -24])), (9, array([104, 104,  40])), (9, array([104, 104, 104])), (10, array([-24, -24, -24])), (10, array([-24, -24,  40])), (10, array([-24, -24, 104])), (10, array([-24,  40, -24])), (10, array([-24,  40,  40])), (10, array([-24,  40, 104])), (10, array([-24, 104, -24])), (10, array([-24, 104,  40])), (10, array([-24, 104, 104])), (10, array([ 40, -24, -24])), (10, array([ 40, -24,  40])), (10, array([ 40, -24, 104])), (10, array([ 40,  40, -24])), (10, array([40, 40, 40])), (10, array([ 40,  40, 104])), (10, array([ 40, 104, -24])), (10, array([ 40, 104,  40])), (10, array([ 40, 104, 104])), (10, array([104, -24, -24])), (10, array([104, -24,  40])), (10, array([104, -24, 104])), (10, array([104,  40, -24])), (10, array([104,  40,  40])), (10, array([104,  40, 104])), (10, array([104, 104, -24])), (10, array([104, 104,  40])), (10, array([104, 104, 104])), (11, array([-24, -24, -24])), (11, array([-24, -24,  40])), (11, array([-24, -24, 104])), (11, array([-24,  40, -24])), (11, array([-24,  40,  40])), (11, array([-24,  40, 104])), (11, array([-24, 104, -24])), (11, array([-24, 104,  40])), (11, array([-24, 104, 104])), (11, array([ 40, -24, -24])), (11, array([ 40, -24,  40])), (11, array([ 40, -24, 104])), (11, array([ 40,  40, -24])), (11, array([40, 40, 40])), (11, array([ 40,  40, 104])), (11, array([ 40, 104, -24])), (11, array([ 40, 104,  40])), (11, array([ 40, 104, 104])), (11, array([104, -24, -24])), (11, array([104, -24,  40])), (11, array([104, -24, 104])), (11, array([104,  40, -24])), (11, array([104,  40,  40])), (11, array([104,  40, 104])), (11, array([104, 104, -24])), (11, array([104, 104,  40])), (11, array([104, 104, 104])), (12, array([-24, -24, -24])), (12, array([-24, -24,  40])), (12, array([-24, -24, 104])), (12, array([-24,  40, -24])), (12, array([-24,  40,  40])), (12, array([-24,  40, 104])), (12, array([-24, 104, -24])), (12, array([-24, 104,  40])), (12, array([-24, 104, 104])), (12, array([ 40, -24, -24])), (12, array([ 40, -24,  40])), (12, array([ 40, -24, 104])), (12, array([ 40,  40, -24])), (12, array([40, 40, 40])), (12, array([ 40,  40, 104])), (12, array([ 40, 104, -24])), (12, array([ 40, 104,  40])), (12, array([ 40, 104, 104])), (12, array([104, -24, -24])), (12, array([104, -24,  40])), (12, array([104, -24, 104])), (12, array([104,  40, -24])), (12, array([104,  40,  40])), (12, array([104,  40, 104])), (12, array([104, 104, -24])), (12, array([104, 104,  40])), (12, array([104, 104, 104])), (13, array([-24, -24, -24])), (13, array([-24, -24,  40])), (13, array([-24, -24, 104])), (13, array([-24,  40, -24])), (13, array([-24,  40,  40])), (13, array([-24,  40, 104])), (13, array([-24, 104, -24])), (13, array([-24, 104,  40])), (13, array([-24, 104, 104])), (13, array([ 40, -24, -24])), (13, array([ 40, -24,  40])), (13, array([ 40, -24, 104])), (13, array([ 40,  40, -24])), (13, array([40, 40, 40])), (13, array([ 40,  40, 104])), (13, array([ 40, 104, -24])), (13, array([ 40, 104,  40])), (13, array([ 40, 104, 104])), (13, array([104, -24, -24])), (13, array([104, -24,  40])), (13, array([104, -24, 104])), (13, array([104,  40, -24])), (13, array([104,  40,  40])), (13, array([104,  40, 104])), (13, array([104, 104, -24])), (13, array([104, 104,  40])), (13, array([104, 104, 104])), (14, array([-24, -24, -24])), (14, array([-24, -24,  40])), (14, array([-24, -24, 104])), (14, array([-24,  40, -24])), (14, array([-24,  40,  40])), (14, array([-24,  40, 104])), (14, array([-24, 104, -24])), (14, array([-24, 104,  40])), (14, array([-24, 104, 104])), (14, array([ 40, -24, -24])), (14, array([ 40, -24,  40])), (14, array([ 40, -24, 104])), (14, array([ 40,  40, -24])), (14, array([40, 40, 40])), (14, array([ 40,  40, 104])), (14, array([ 40, 104, -24])), (14, array([ 40, 104,  40])), (14, array([ 40, 104, 104])), (14, array([104, -24, -24])), (14, array([104, -24,  40])), (14, array([104, -24, 104])), (14, array([104,  40, -24])), (14, array([104,  40,  40])), (14, array([104,  40, 104])), (14, array([104, 104, -24])), (14, array([104, 104,  40])), (14, array([104, 104, 104])), (15, array([-24, -24, -24])), (15, array([-24, -24,  40])), (15, array([-24, -24, 104])), (15, array([-24,  40, -24])), (15, array([-24,  40,  40])), (15, array([-24,  40, 104])), (15, array([-24, 104, -24])), (15, array([-24, 104,  40])), (15, array([-24, 104, 104])), (15, array([ 40, -24, -24])), (15, array([ 40, -24,  40])), (15, array([ 40, -24, 104])), (15, array([ 40,  40, -24])), (15, array([40, 40, 40])), (15, array([ 40,  40, 104])), (15, array([ 40, 104, -24])), (15, array([ 40, 104,  40])), (15, array([ 40, 104, 104])), (15, array([104, -24, -24])), (15, array([104, -24,  40])), (15, array([104, -24, 104])), (15, array([104,  40, -24])), (15, array([104,  40,  40])), (15, array([104,  40, 104])), (15, array([104, 104, -24])), (15, array([104, 104,  40])), (15, array([104, 104, 104])), (16, array([-24, -24, -24])), (16, array([-24, -24,  40])), (16, array([-24, -24, 104])), (16, array([-24,  40, -24])), (16, array([-24,  40,  40])), (16, array([-24,  40, 104])), (16, array([-24, 104, -24])), (16, array([-24, 104,  40])), (16, array([-24, 104, 104])), (16, array([ 40, -24, -24])), (16, array([ 40, -24,  40])), (16, array([ 40, -24, 104])), (16, array([ 40,  40, -24])), (16, array([40, 40, 40])), (16, array([ 40,  40, 104])), (16, array([ 40, 104, -24])), (16, array([ 40, 104,  40])), (16, array([ 40, 104, 104])), (16, array([104, -24, -24])), (16, array([104, -24,  40])), (16, array([104, -24, 104])), (16, array([104,  40, -24])), (16, array([104,  40,  40])), (16, array([104,  40, 104])), (16, array([104, 104, -24])), (16, array([104, 104,  40])), (16, array([104, 104, 104])), (17, array([-24, -24, -24])), (17, array([-24, -24,  40])), (17, array([-24, -24, 104])), (17, array([-24,  40, -24])), (17, array([-24,  40,  40])), (17, array([-24,  40, 104])), (17, array([-24, 104, -24])), (17, array([-24, 104,  40])), (17, array([-24, 104, 104])), (17, array([ 40, -24, -24])), (17, array([ 40, -24,  40])), (17, array([ 40, -24, 104])), (17, array([ 40,  40, -24])), (17, array([40, 40, 40])), (17, array([ 40,  40, 104])), (17, array([ 40, 104, -24])), (17, array([ 40, 104,  40])), (17, array([ 40, 104, 104])), (17, array([104, -24, -24])), (17, array([104, -24,  40])), (17, array([104, -24, 104])), (17, array([104,  40, -24])), (17, array([104,  40,  40])), (17, array([104,  40, 104])), (17, array([104, 104, -24])), (17, array([104, 104,  40])), (17, array([104, 104, 104])), (18, array([-24, -24, -24])), (18, array([-24, -24,  40])), (18, array([-24, -24, 104])), (18, array([-24,  40, -24])), (18, array([-24,  40,  40])), (18, array([-24,  40, 104])), (18, array([-24, 104, -24])), (18, array([-24, 104,  40])), (18, array([-24, 104, 104])), (18, array([ 40, -24, -24])), (18, array([ 40, -24,  40])), (18, array([ 40, -24, 104])), (18, array([ 40,  40, -24])), (18, array([40, 40, 40])), (18, array([ 40,  40, 104])), (18, array([ 40, 104, -24])), (18, array([ 40, 104,  40])), (18, array([ 40, 104, 104])), (18, array([104, -24, -24])), (18, array([104, -24,  40])), (18, array([104, -24, 104])), (18, array([104,  40, -24])), (18, array([104,  40,  40])), (18, array([104,  40, 104])), (18, array([104, 104, -24])), (18, array([104, 104,  40])), (18, array([104, 104, 104])), (19, array([-24, -24, -24])), (19, array([-24, -24,  40])), (19, array([-24, -24, 104])), (19, array([-24,  40, -24])), (19, array([-24,  40,  40])), (19, array([-24,  40, 104])), (19, array([-24, 104, -24])), (19, array([-24, 104,  40])), (19, array([-24, 104, 104])), (19, array([ 40, -24, -24])), (19, array([ 40, -24,  40])), (19, array([ 40, -24, 104])), (19, array([ 40,  40, -24])), (19, array([40, 40, 40])), (19, array([ 40,  40, 104])), (19, array([ 40, 104, -24])), (19, array([ 40, 104,  40])), (19, array([ 40, 104, 104])), (19, array([104, -24, -24])), (19, array([104, -24,  40])), (19, array([104, -24, 104])), (19, array([104,  40, -24])), (19, array([104,  40,  40])), (19, array([104,  40, 104])), (19, array([104, 104, -24])), (19, array([104, 104,  40])), (19, array([104, 104, 104])), (20, array([-24, -24, -24])), (20, array([-24, -24,  40])), (20, array([-24, -24, 104])), (20, array([-24,  40, -24])), (20, array([-24,  40,  40])), (20, array([-24,  40, 104])), (20, array([-24, 104, -24])), (20, array([-24, 104,  40])), (20, array([-24, 104, 104])), (20, array([ 40, -24, -24])), (20, array([ 40, -24,  40])), (20, array([ 40, -24, 104])), (20, array([ 40,  40, -24])), (20, array([40, 40, 40])), (20, array([ 40,  40, 104])), (20, array([ 40, 104, -24])), (20, array([ 40, 104,  40])), (20, array([ 40, 104, 104])), (20, array([104, -24, -24])), (20, array([104, -24,  40])), (20, array([104, -24, 104])), (20, array([104,  40, -24])), (20, array([104,  40,  40])), (20, array([104,  40, 104])), (20, array([104, 104, -24])), (20, array([104, 104,  40])), (20, array([104, 104, 104])), (21, array([-24, -24, -24])), (21, array([-24, -24,  40])), (21, array([-24, -24, 104])), (21, array([-24,  40, -24])), (21, array([-24,  40,  40])), (21, array([-24,  40, 104])), (21, array([-24, 104, -24])), (21, array([-24, 104,  40])), (21, array([-24, 104, 104])), (21, array([ 40, -24, -24])), (21, array([ 40, -24,  40])), (21, array([ 40, -24, 104])), (21, array([ 40,  40, -24])), (21, array([40, 40, 40])), (21, array([ 40,  40, 104])), (21, array([ 40, 104, -24])), (21, array([ 40, 104,  40])), (21, array([ 40, 104, 104])), (21, array([104, -24, -24])), (21, array([104, -24,  40])), (21, array([104, -24, 104])), (21, array([104,  40, -24])), (21, array([104,  40,  40])), (21, array([104,  40, 104])), (21, array([104, 104, -24])), (21, array([104, 104,  40])), (21, array([104, 104, 104])), (22, array([-24, -24, -24])), (22, array([-24, -24,  40])), (22, array([-24, -24, 104])), (22, array([-24,  40, -24])), (22, array([-24,  40,  40])), (22, array([-24,  40, 104])), (22, array([-24, 104, -24])), (22, array([-24, 104,  40])), (22, array([-24, 104, 104])), (22, array([ 40, -24, -24])), (22, array([ 40, -24,  40])), (22, array([ 40, -24, 104])), (22, array([ 40,  40, -24])), (22, array([40, 40, 40])), (22, array([ 40,  40, 104])), (22, array([ 40, 104, -24])), (22, array([ 40, 104,  40])), (22, array([ 40, 104, 104])), (22, array([104, -24, -24])), (22, array([104, -24,  40])), (22, array([104, -24, 104])), (22, array([104,  40, -24])), (22, array([104,  40,  40])), (22, array([104,  40, 104])), (22, array([104, 104, -24])), (22, array([104, 104,  40])), (22, array([104, 104, 104])), (23, array([-24, -24, -24])), (23, array([-24, -24,  40])), (23, array([-24, -24, 104])), (23, array([-24,  40, -24])), (23, array([-24,  40,  40])), (23, array([-24,  40, 104])), (23, array([-24, 104, -24])), (23, array([-24, 104,  40])), (23, array([-24, 104, 104])), (23, array([ 40, -24, -24])), (23, array([ 40, -24,  40])), (23, array([ 40, -24, 104])), (23, array([ 40,  40, -24])), (23, array([40, 40, 40])), (23, array([ 40,  40, 104])), (23, array([ 40, 104, -24])), (23, array([ 40, 104,  40])), (23, array([ 40, 104, 104])), (23, array([104, -24, -24])), (23, array([104, -24,  40])), (23, array([104, -24, 104])), (23, array([104,  40, -24])), (23, array([104,  40,  40])), (23, array([104,  40, 104])), (23, array([104, 104, -24])), (23, array([104, 104,  40])), (23, array([104, 104, 104]))]
648
```
可以看出先当与在train_list（30*0.8=24）中每遍历一次，就增加了27个patch的索引，所以最后一共24*27=648个索引  
最后返回这个patch_index

### compute_patch_indices
```
def compute_patch_indices(image_shape, patch_size, overlap, start=None):
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
```
我们这里分析最简单的情况，没有任何的可选功能的情况下  
start = None    
overlap=0   
让我们看看函数是怎么运行的
```
overlap=0
if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))
print(overlap)
print((patch_size - overlap))
#输出
[0 0 0]
[64 64 64]
```
isinstance() 函数来判断一个对象是否是一个已知的类型  
并将overlap形式转化
```
print(len(image_shape))
print([0]*len(image_shape))
#输出
3
[0, 0, 0]
```
然后执行（默认overlap=0）    
```  
if start is None:
    n_patches = np.ceil(image_shape / (patch_size ))
    overflow = (patch_size - overlap) * n_patches - image_shape
    start = -np.ceil(overflow/2)  
```
也是不好理解哦，一句句看  
ceil() 函数返回数字的上入整数。
```
n_patches = np.ceil(image_shape / (patch_size - overlap))
print(n_patches)
#输出为
[3. 3. 3.]#144/64向上取整
```
然后是
```
overflow = (patch_size - overlap) * n_patches - image_shape
print(overflow)
#输出
[48. 48. 48.]
```
用之前得到的n_patches乘以patch_size 可以算出我们超出了原图像的范围  
因为之前计算n_patches时用到了向上取整，所以范围一定会大  
3组patch*每组大小64=192比原图144大了48个像素点  
```
start = -np.ceil(overflow/2)
print(start)
#输出
[-24. -24. -24.]
```
最后算出start的值  

stop = image_shape = data_file.root.data.shape[-3:]
step = patch_size #(64,64,64)  
我们这里可以看一下stop的值
```
data_file=data_file_opened
image_shape = data_file.root.data.shape[-3:]
print(image_shape)
#输出
(144, 144, 144)
```
最后执行get_set_of_patch_indices(start, stop, step)来得到最终的索引列表，函数详解见下文


#### get_set_of_patch_indices
```
def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)
```
虽然这个函数看起来就一行，但要是第一次见确实相当的难理解  
[python笔记：numpy中mgrid的用法](https://blog.csdn.net/abc13526222160/article/details/88559162)  
[np.mgrid的用法](https://www.cnblogs.com/wanghui-garcia/p/10763103.html)  
对于mgrid的用法，先看看以上两篇文章  
现在我们再来分析这个函数
```
table_inx=np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],start[2]:stop[2]:step[2]]
print(table_inx)
#输出
[[[[-24. -24. -24.]
   [-24. -24. -24.]
   [-24. -24. -24.]]

  [[ 40.  40.  40.]
   [ 40.  40.  40.]
   [ 40.  40.  40.]]

  [[104. 104. 104.]
   [104. 104. 104.]
   [104. 104. 104.]]]


 [[[-24. -24. -24.]
   [ 40.  40.  40.]
   [104. 104. 104.]]

  [[-24. -24. -24.]
   [ 40.  40.  40.]
   [104. 104. 104.]]

  [[-24. -24. -24.]
   [ 40.  40.  40.]
   [104. 104. 104.]]]


 [[[-24.  40. 104.]
   [-24.  40. 104.]
   [-24.  40. 104.]]

  [[-24.  40. 104.]
   [-24.  40. 104.]
   [-24.  40. 104.]]

  [[-24.  40. 104.]
   [-24.  40. 104.]
   [-24.  40. 104.]]]]
```
```
print(table_inx.shape)
a=(np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int))
print(a)
print(a.shape)
#输出
(3, 3, 3, 3)
[[-24 -24 -24]
 [-24 -24  40]
 [-24 -24 104]
 [-24  40 -24]
 [-24  40  40]
 [-24  40 104]
 [-24 104 -24]
 [-24 104  40]
 [-24 104 104]
 [ 40 -24 -24]
 [ 40 -24  40]
 [ 40 -24 104]
 [ 40  40 -24]
 [ 40  40  40]
 [ 40  40 104]
 [ 40 104 -24]
 [ 40 104  40]
 [ 40 104 104]
 [104 -24 -24]
 [104 -24  40]
 [104 -24 104]
 [104  40 -24]
 [104  40  40]
 [104  40 104]
 [104 104 -24]
 [104 104  40]
 [104 104 104]]
(27, 3)
```
可以看到我们最终得到的是一个索引列表，记录了所有patch块的起始位置  
每行3个乘以每列3个乘以深度3个=27个    

## add_data
```
def add_data(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=False, skip_blank=True, permute=False):
    """
    Adds data from the data file to the given lists of feature and target data
    将数据文件中的数据添加到要素和目标数据的给定列表中
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
    if augment:
        if patch_shape is not None:
            affine = data_file.root.affine[index[0]]
        else:
            affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)
```
此函数的作用是对之前确定好的一组训练集或验证集数据进行切片，来产生每组数据的patch数据及其label。  
要将data_file中的数据将附加到的数据列表x_list。  
要将data_file中的目标数据将附加到的数据列表y_list。
这里为了便于理解，所有数据增强的参数均设置为0.    
那么其实就只执行了
```
data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
if not skip_blank or np.any(truth != 0):
    x_list.append(data)
    y_list.append(truth)
```
利用get_data_from_file函数从文件中读取patch数据，详见下文  
并把得到的数据与标签加到x_list与y_list中

### get_data_from_file
```
def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file.root.data[index], data_file.root.truth[index, 0]
    return x, y
```
首先我们注意到一点，其在定义中调用了本身  
data, truth = get_data_from_file(data_file, index, patch_shape=None)
这其实和我们之前学过的递归函数的原理是一样的  
我们执行到这一句时，patch_shape=None，所以会执行else的语句  
看一下他是怎么运行的  
```
index1, patch_index = index
print(index1)
print(patch_index)
#输出
23
[104 104 104]
```
然后在执行这一句之前  
x, y = data_file.root.data[index], data_file.root.truth[index, 0]  
我们先看一下之前在data.py中提到过的，储存图片数据与标签数据的可扩展压缩数组  
```
print(np.asarray(data_storage).shape)
print(np.asarray(truth_storage).shape)
#输出
(30, 4, 144, 144, 144)
(30, 1, 144, 144, 144)
```
然后，我们再来看  
x, y = data_file.root.data[index], data_file.root.truth[index, 0]    
因为data_file.root.data是五维的，每组数据在第一维上叠加，所以data_file.root.data[index]就代表第23组数据的四种模态图片。  
而data_file.root.truth也是五维的，但其实我们每组数据只有一张标签图像，所以为data_file.root.truth[index, 0]   
我们看一下输出来验证一下  
```
x, y = data_file.root.data[index1], data_file.root.truth[index1, 0]
print(x)
print(x.shape)
#输出
[[[[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  ...

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]]


 [[[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  ...

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]]


 [[[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  ...

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]]


 [[[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  ...

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]

  [[0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   ...
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]
   [0. 0. 0. ... 0. 0. 0.]]]]
(4, 144, 144, 144)
```
```
print(y)
print(y.shape)
#输出
[[[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 ...

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]

 [[0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  ...
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]
  [0 0 0 ... 0 0 0]]]
(144, 144, 144)
```
然后该执行get_patch_from_3d_data函数获取x，y了，详见下文  
最后返回x，y  

#### get_patch_from_3d_data
```
def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]
```
从numpy数组返回一个patch。  
data：从中获取patch的numpy数组。
patch_shape：补丁的形状/大小。  
patch_index：补丁的角索引。  
return：从numpy数组从获取具有指定补丁形状的数据。

先看前三句  
```
patch_index = np.asarray(patch_index, dtype=np.int16)
patch_shape = np.asarray((64,64,64))
print(patch_index)
print(patch_shape)
image_shape = data.shape[-3:]
print(image_shape)
#输出
[104 104 104]
[64 64 64]
(144, 144, 144)
```
然后要对超出图片范围的部分进行操作  
```
if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
    data, patch_index = fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)
```
因为我们之前在进行确定patch个数的时候采取了向上取整，所以我们的索引坐标会超出图片的总大小  
比如现在我们正在执行的，104+64>144,同样的我们一些patch开始的索引-24也小于图片的初始索引0    
```
return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
            patch_index[2]:patch_index[2]+patch_shape[2]]
```
最后获取对应索引位置patch立方体的数据
```
a=data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]
print(a)
print(a.shape)
#输出
[[[[ 974.64435  967.2424   959.2515  ...    0.         0.
       0.     ]
   [ 990.2376   982.45245  976.0181  ...    0.         0.
       0.     ]
   [1014.01    1011.8526  1012.1444  ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 979.8702   969.1078   957.12286 ...    0.         0.
       0.     ]
   [1019.1183  1006.22644  993.1565  ...    0.         0.
       0.     ]
   [1042.1243  1035.1311  1029.3956  ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 999.7032   988.7418   977.0992  ...    0.         0.
       0.     ]
   [1035.4014  1024.9207  1013.7255  ...    0.         0.
       0.     ]
   [1039.5989  1035.5583  1031.2466  ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  ...

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]]


 [[[ 144.65683  165.30124  180.96454 ...    0.         0.
       0.     ]
   [ 153.01872  175.63034  187.93913 ...    0.         0.
       0.     ]
   [ 162.61511  174.65828  179.1287  ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 157.46268  171.40187  180.69766 ...    0.         0.
       0.     ]
   [ 168.83534  178.6118   181.83922 ...    0.         0.
       0.     ]
   [ 177.60065  184.03499  185.06255 ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 166.20416  171.03703  177.93298 ...    0.         0.
       0.     ]
   [ 176.08592  174.61232  177.79686 ...    0.         0.
       0.     ]
   [ 177.65994  175.62888  178.42007 ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  ...

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]]


 [[[ 458.       463.       467.      ...    0.         0.
       0.     ]
   [ 449.       451.       452.      ...    0.         0.
       0.     ]
   [ 432.       437.       443.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 467.       471.       476.      ...    0.         0.
       0.     ]
   [ 447.       448.       449.      ...    0.         0.
       0.     ]
   [ 438.       440.       443.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 450.       453.       455.      ...    0.         0.
       0.     ]
   [ 432.       432.       433.      ...    0.         0.
       0.     ]
   [ 440.       435.       430.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  ...

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]]


 [[[ 398.2283   405.59125  413.5954  ...    0.         0.
       0.     ]
   [ 404.69037  413.0081   421.04782 ...    0.         0.
       0.     ]
   [ 392.00458  400.79367  409.79398 ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 385.11923  391.67032  398.71704 ...    0.         0.
       0.     ]
   [ 393.11557  400.79193  409.13693 ...    0.         0.
       0.     ]
   [ 390.004    398.76068  406.39658 ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[ 379.12677  385.17493  390.83636 ...    0.         0.
       0.     ]
   [ 391.53232  397.18793  402.48724 ...    0.         0.
       0.     ]
   [ 393.9321   399.31738  403.58066 ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  ...

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]

  [[   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   ...
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]
   [   0.         0.         0.      ...    0.         0.
       0.     ]]]]
(4, 64, 64, 64)
```

##### fix_out_of_bound_patch_attempt
```
def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index
```
填充数据并更改patch索引，以使patch正确无误。  
由于数据和label的图片维度不一样  
用image_shape = data.shape[-3:]来获取单张图片大小  

如果是索引小于0的情况需要前面补，补的个数相当于patch_index的值  
如果是索引+patch之后大于图片范围，则后面补，补的个数等于超出图片的范围  
pad_before = np.abs((patch_index < 0) * patch_index)  
pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))  
然后用np.stack([pad_before, pad_after], axis=1)把之前之后的数串起来  
stack()函数的意思，就是对arrays里面的每个元素(可能是个列表，元组，或者是个numpy的数组)变成numpy的数组后，再对每个元素增加一维(至于维度加在哪里，是靠axis控制的)，然后再把这些元素串起来  
```
pad_args = np.stack([pad_before, pad_after], axis=1)
print(pad_args)
#输出
[[ 0 24]
 [ 0 24]
 [ 0 24]]
```
```
if pad_args.shape[0] < len(data.shape):
    pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
```
这里判断一下我们的输入数据data是数据还是标签，pad_args.shape[0]=3，如果为数据的话，data.shape=4，则执行下面一句话改变一下pad_args的形式
```
pad_args = [[0, 0]] * (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
print(pad_args)
#输出
[[0, 0], [0, 24], [0, 24], [0, 24]]
```
```
data = np.pad(data, pad_args, mode="edge")
patch_index += pad_before
return data, patch_index
```
用pad函数在图片周围填充一定量的像素点,‘edge’——表示用边缘值填充  
patch_index += pad_before的作用是如果开始的index值为负的话，对于填充完的图像数据，现在索引应该从零开始  
如果开始的index是正的比如144，我们在后边填充完，其index没有改变，还是144  

## convert_data
```
def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y
```
主要是判断一下标签个数，如果我们只有一种标签，那么就是说我们标签图像中的值为0个1就够了  
但如果含有多个标签，比如说我们brats的数据有1，2，4三种标签，就要进行额外操作了，get_multi_class_labels将标签图转换为一组二进制numpy数组，shape: (n_samples, n_labels, ...)  
最后返回x，y

# get_number_of_steps
```
def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1
```
产生训练集的steps_per_epoch及验证集的validation_steps，给fit_generator函数使用。  
之前get_number_of_patches函数返回的是n_samples=648，远大于训练集batch_size=6  
用remainder取余函数判断，如果n_samples对batch_size余数为零的情况下，就直接用//除  
若不为零，则用//向下取整，再加一  

# get_number_of_patches
```
def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True):
    if patch_shape:
        index_list = create_patch_index_list(index_list, data_file.root.data.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        count = 0
        for index in index_list:
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
            if len(x_list) > 0:
                count += 1
        return count
    else:
        return len(index_list)
```
这里考虑patch_shape不为None的情况，  
首先用create_patch_index_list产生patch的索引列表，这个函数之前读取数据的时候用过，下面讲的有   
在函数的讲解中，我们也能看到，len(patch_index)=27*24=648个索引  
所以，在这里，在我们不跳过空白数据的情况下，遍历648个索引点，最终count=648
