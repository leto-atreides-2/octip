3D U-Net源码解析之data.py
===============

本文主要介绍之前在train.py中提到的函数

<!-- TOC -->

- [write_data_to_file](#write_data_to_file)
  - [参数](#参数)
  - [第一块代码](#第一块代码)
  - [第二块代码](#第二块代码)
  - [第三块代码](#第三块代码)
  - [第四块代码](#第四块代码)
  - [第五块代码](#第五块代码)
- [create_data_file](#create_data_file)
- [write_image_data_to_file](#write_image_data_to_file)
  - [add_data_to_storage](#add_data_to_storage)

<!-- /TOC -->

这个文件相当的复杂，写起来也比较乱，希望大家在观看的时候能够根据函数的调用情况并结合我的目录，跳到相应的位置（子函数）看，不然很容易忘了自己在干啥QAQ


# write_data_to_file
write_data_to_file 功能是接收一组训练图像并将这些图像写入hdf5文件

```
def write_data_to_file(training_data_files, out_file, image_shape, truth_dtype=np.uint8, subject_ids=None,
                       normalize=True, crop=True):
    n_samples = len(training_data_files)
    n_channels = len(training_data_files[0]) - 1

    try:
        hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                                  n_channels=n_channels,
                                                                                  n_samples=n_samples,
                                                                                  image_shape=image_shape)
    except Exception as e:
        # If something goes wrong, delete the incomplete data file
        os.remove(out_file)
        raise e

    write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                             truth_dtype=truth_dtype, n_channels=n_channels, affine_storage=affine_storage, crop=crop)
    if subject_ids:
        hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
    if normalize:
        normalize_data_storage(data_storage)
    hdf5_file.close()
    return out_file
```

## 参数
1. training_data_files   
training_data_files包含训练数据文件的元组tuple列表。 在每个元组tuple中，几种模式应该以相同的顺序列出。 每个元组中的最后一项必须是带标签的图像（truth）。可以在我之前写的train.py解析中观察这个参数。例如：    
  [('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-truth.nii.gz'), ('sub2-T1.nii.gz', 'sub2-T2.nii.gz', 'sub2-truth.nii.gz')]
2. out_file：hdf5文件写入的位置
3. image_shape：需要存进hdf5文件中图像的大小
4. truth_dtype：默认是8位无符号整数
5. 函数返回：写入了图像数据的hdf5文件的位置

## 第一块代码
```
n_samples = len(training_data_files)
n_channels = len(training_data_files[0]) - 1
```
在上一篇分析train.py时我们看过，len(training_data_files)代表是preprocessed文件中所有所有图像文件夹的个数，即样本数。每个文件夹的储存为一个元组，而每个元组中有5种形式（"t1", "t1ce", "flair", "t2"+"truth"）的nii文件，而且每个元组中图片排列的顺序都一样。所以n_channels = len(training_data_files[0]) - 1为四种形式的训练集nii文件，即channels数。我们可以看一下其输出
```
training_files = fetch_training_data_files()
print(type(training_files))
print(len(training_files))
print(len(training_files[0]) - 1)
# 输出
<class 'list'>
30
4
```
## 第二块代码
```
try:
    hdf5_file, data_storage, truth_storage, affine_storage = create_data_file(out_file,
                                                                              n_channels=n_channels,
                                                                              n_samples=n_samples,
                                                                              image_shape=image_shape)
except Exception as e:
    # If something goes wrong, delete the incomplete data file
    os.remove(out_file)
    raise e
```
create_data_file详解见下文，产生了四个输出  table:df5_file，以及三个可扩展的压缩数组
这里使用try...except...程序结构来获取异常（Exception）信息，可以有助于快速定位有错误程序语句的位置。
而且，如果出现问题，通过os.remove(out_file)删除不完整的数据文件

## 第三块代码
```
write_image_data_to_file(training_data_files, data_storage, truth_storage, image_shape,
                           truth_dtype=truth_dtype, n_channels=n_channels, affine_storage=affine_storage, crop=crop)
```
函数write_image_data_to_file()的作用是向之前创建的压缩可扩展的数组中写入图像数据。
这一块虽然看起来就一行命令，但其涉及到非常多的子函数，我在下面对write_image_data_to_file函数及其涉及到的子函数做了详细的讲解，希望大家跳到下面write_image_data_to_file函数观看。
## 第四块代码
```
if subject_ids:
    hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
if normalize:
    normalize_data_storage(data_storage)
```
subject_ids我运行时没有用到，这里先不研究。
讲一下normalize，其定义为
```
def normalize_data(data, mean, std):
    #data：[4,144,144,144]
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    #[n_example,4,144,144,144]
    for index in range(data_storage.shape[0]):
        #[4,144,144,144]
        data = data_storage[index]
        #分别求出每个模态的均值和标准差
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    #求每个模态在所有样本上的均值和标准差[n_example,4]==>[4]
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        #根据均值和标准差对每一个样本归一化
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage
```
作用是对训练集所有模态image归一化
## 第五块代码
```
hdf5_file.close()
return out_file
```
呜呜呜，这个函数终于结束了，临走前记得关闭文件哦~

# create_data_file
```
def create_data_file(out_file, n_channels, n_samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, n_channels] + list(image_shape))
    truth_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=n_samples)
    truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=n_samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                             filters=filters, expectedrows=n_samples)
    return hdf5_file, data_storage, truth_storage, affine_storage
```
这里用到了好多Python Tables 的知识，大家可以参考 [Python Tables 学习笔记](https://blog.csdn.net/lengyuexiang123/article/details/53558779)
```
hdf5_file = tables.open_file(out_file, mode='w')
```
新建一个hdf5文件，文件名是out_file， 写的模式。
```
filters = tables.Filters(complevel=5, complib='blosc')#声明压缩类型及深度
data_shape = tuple([0, n_channels] + list(image_shape))
truth_shape = tuple([0, 1] + list(image_shape))
data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                       filters=filters, expectedrows=n_samples)
truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                        filters=filters, expectedrows=n_samples)
affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                         filters=filters, expectedrows=n_samples)
```

压缩数组（Compression Array）
HDF文件还可以进行压缩存储，压缩方式有blosc, zlib, 和 lzo。Zlib和lzo压缩方式需要另外的包，blosc在tables是自带的。我们需要定义一个filter来说明压缩方式及压缩深度。另外，我们使用creatCArray来创建压缩矩阵。

压缩可扩展数组（Compression & Enlargeable Array）
压缩数组，初始化之后就不能改变大小，但现实很多时候，我们只知道维度，并不知道我们数据的长度。这个时候，我们需要这个数组是可以扩展的。HDF文件也提供这样的接口，我们能够扩展其一个维度。同CArray一样，我们也先要定filter来声明压缩类型及深度。最重要的是，我们把可以扩展这个维度的shape设置为0。这里写0，代表这个维度是可拓展的。
所以，我们可以观察一下我们数据的维度
```
hdf5_file = tables.open_file(config["data_file"], mode='w')
filters = tables.Filters(complevel=5, complib='blosc')
data_shape = tuple([0, n_channels] + list(config["image_shape"]))
truth_shape = tuple([0, 1] + list(config["image_shape"]))
print(data_shape)
print(truth_shape )
# 输出
(0, 4, 144, 144, 144)
(0, 1, 144, 144, 144)
```
可以看到，为了压缩可扩展数组，我们对数据的shape进行了调整，训练集数据因为有四种模式所以有四个channels
```
data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                         filters=filters, expectedrows=n_samples)
truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                          filters=filters, expectedrows=n_samples)
affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float32Atom(), shape=(0, 4, 4),
                                           filters=filters, expectedrows=n_samples)
return hdf5_file, data_storage, truth_storage, affine_storage
  ```
这个create_earray就是创建可拓展矩阵的函数（Enlargeable）。
返回四个输出   table:df5_file，以及三个可扩展的压缩数组

# write_image_data_to_file
```
def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage,
                             truth_dtype=np.uint8, crop=True):
    for set_of_files in image_files:
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        subject_data = [image.get_data() for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                            truth_dtype)
    return data_storage, truth_storage
```
函数write_image_data_to_file()的作用是向之前创建的压缩可扩展的数组中写入图像数据．
```
for set_of_files in image_files:
```
遍历之前用fetch_training_data_files()获得的所有子文件夹的路径。不同模态图像路径的元组('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz')
```
images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
```
对4个模态+truth图像根据前景背景裁剪  
reslice_image_set函数定义如下
```
    def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False):
        #in_files:('sub1-T1.nii.gz', 'sub1-T2.nii.gz', 'sub1-flair.nii.gz','sub1-t1ce.nii.gz','sub1-truth.nii.gz')
        #label_indices:模态个数-4
        #对图像进行裁剪
        if crop:
            #返回各个维度要裁剪的范围[slice(),slice(),slice()]
            crop_slices = get_cropping_parameters([in_files])
        else:
            crop_slices = None
        #对in_files中的每个image裁剪放缩后返回的image列表
        images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
        if out_files:
            for image, out_file in zip(images, out_files):
                image.to_filename(out_file)
            return [os.path.abspath(out_file) for out_file in out_files]
        else:
            return images
```
```
subject_data = [image.get_data() for image in images]
```
获取4个模态+truth的image的数组
```
add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                        truth_dtype)
```
添加1份subject_data数据，写入时将subject_data扩展到与create_data_file中定义的维度相同，并完成对可扩展数组的写入
希望大家转到下面的add_data_to_storage看，写的比较详细
```
return data_storage, truth_storage
```
读取并写入完所有的图片后，返回训练集和标签的可扩展数组

## add_data_to_storage
```

def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):
    #添加1份subject_data数据，写入时将subject_data扩展到与create_data_file中定义的维度相同
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])#np.asarray:==>[4,144,144,144] 扩展=new.axis:[1,4,144,144,144]
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])#np.asarray:==>[144,144,144] 扩展=new.axis:[1,1,144,144,144]
    affine_storage.append(np.asarray(affine)[np.newaxis])#np.asarray:==>[4,4] 扩展=new.axis:[1,4,,4]
```
这个函数也是看的让人有点晕的，其实就是我们上一步得到所有的图像数据之后subject_data将其分成训练数据和标签数据分别加入到我们已经创建好的可扩展数组data_storage与truth_storage中，但是现在有一个问题，我们subject_data得到的图像的数据与我们之前定义好的可扩展数组的shape不一样，无法直接用append叠加，现在我们要做的就是分好数据并改变它们的shape以便使用append写入可扩展数组中
这里np.newaxis的功能为插入新维度，看起来比较乱，我们来举个例子研究一下：
```
array=np.arange(40)
print(array)
print(array[:2])
array=array.reshape(5,2,2,2)
print(array)
#输出
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
[0 1]
[[[[ 0  1]
   [ 2  3]]

  [[ 4  5]
   [ 6  7]]]


 [[[ 8  9]
   [10 11]]

  [[12 13]
   [14 15]]]


 [[[16 17]
   [18 19]]

  [[20 21]
   [22 23]]]


 [[[24 25]
   [26 27]]

  [[28 29]
   [30 31]]]


 [[[32 33]
   [34 35]]

  [[36 37]
   [38 39]]]]
```
这里我选40个数，将其分成(5,2,2,2)的形式，这其实和我们的subject_data形式是一样的，第一维（shape[0]）=5，即训练集的四种形态和一个truth。
```
data_for_train = array[:4]
print(data_for_train)
print(data_for_train.shape)
#输出
[[[[ 0  1]
   [ 2  3]]

  [[ 4  5]
   [ 6  7]]]


 [[[ 8  9]
   [10 11]]

  [[12 13]
   [14 15]]]


 [[[16 17]
   [18 19]]

  [[20 21]
   [22 23]]]


 [[[24 25]
   [26 27]]

  [[28 29]
   [30 31]]]]
(4, 2, 2, 2)
```
用array[:4]划分出我们的训练集，shape=(4, 2, 2, 2)为4种形态的(2, 2, 2)的图像数据
```
data_for_truth = array[4]
print(data_for_truth)
print(data_for_truth.shape)
#输出
[[[32 33]
  [34 35]]

 [[36 37]
  [38 39]]]
(2, 2, 2)
```
array[4]是我们最后的标签图像，因为只是一张图像所以shape=(2, 2, 2)
```
data_for_train=data_for_train[np.newaxis]
print(data_for_train)
print(data_for_train.shape)
#输出
[[[[[ 0  1]
    [ 2  3]]

   [[ 4  5]
    [ 6  7]]]


  [[[ 8  9]
    [10 11]]

   [[12 13]
    [14 15]]]


  [[[16 17]
    [18 19]]

   [[20 21]
    [22 23]]]


  [[[24 25]
    [26 27]]

   [[28 29]
    [30 31]]]]]
(1, 4, 2, 2, 2)
```
用[np.newaxis]来增加一维，变成和可扩展数组一样的形式(1, 4, 2, 2, 2)
```
data_for_truth=data_for_truth[np.newaxis][np.newaxis]
print(data_for_truth)
print(data_for_truth.shape)
#输出
[[[[[32 33]
    [34 35]]

   [[36 37]
    [38 39]]]]]
(1, 1, 2, 2, 2)
```
用两次np.newaxis]来增加两维，变成和可扩展数组一样的形式(1, 1, 2, 2, 2)

通过这个例子，大家应该就明白了这个函数的原理，让我们来看看函数的输出
```
def add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, affine, n_channels, truth_dtype):
    print('data_storage变化:')
    print((np.asarray(subject_data[:n_channels])).shape)
    data_storage.append(np.asarray(subject_data[:n_channels])[np.newaxis])
    print((np.asarray(subject_data[:n_channels])[np.newaxis]).shape)
    print('truth_storage变化:')
    print((np.asarray(subject_data[n_channels])).shape)
    truth_storage.append(np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis])
    print((np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis]).shape)
    print((np.asarray(subject_data[n_channels], dtype=truth_dtype)[np.newaxis][np.newaxis]).shape)
    affine_storage.append(np.asarray(affine)[np.newaxis])
```
```
def write_image_data_to_file(image_files, data_storage, truth_storage, image_shape, n_channels, affine_storage,
                             truth_dtype=np.uint8, crop=True):
    for set_of_files in image_files:
        images = reslice_image_set(set_of_files, image_shape, label_indices=len(set_of_files) - 1, crop=crop)
        subject_data = [image.get_data() for image in images]
        add_data_to_storage(data_storage, truth_storage, affine_storage, subject_data, images[0].affine, n_channels,
                            truth_dtype)
    return data_storage, truth_storage
write_image_data_to_file(training_files, data_storage, truth_storage, image_shape=(144, 144, 144),
                                 truth_dtype=np.uint8, n_channels=n_channels, affine_storage=affine_storage, crop=True)
```
输出为
```
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0006/t1ce.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0006/flair.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0006/t2.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0006/truth.nii.gz
data_storage变化:
(4, 144, 144, 144)
(1, 4, 144, 144, 144)
truth_storage变化:
(144, 144, 144)
(1, 144, 144, 144)
(1, 1, 144, 144, 144)
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/t1.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/t1ce.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/flair.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/t2.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/truth.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/t1.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/t1ce.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/flair.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/t2.nii.gz
Reading: data/preprocessed/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/TCGA-02-0033/truth.nii.gz
data_storage变化:
(4, 144, 144, 144)
(1, 4, 144, 144, 144)
truth_storage变化:
(144, 144, 144)
(1, 144, 144, 144)
(1, 1, 144, 144, 144)
```
可以看到和我们的例子其实一样，再看看我们可扩展数组的输出
```
print(np.asarray(data_storage).shape)
print(np.asarray(truth_storage).shape)
#输出
(5, 4, 144, 144, 144)
(5, 1, 144, 144, 144)
```
可以看出，5次操作后，数据在第一维上叠加了5次，这样就实现了可扩展数组的写入
