3D UNet for classification of OCT volume
=============================================
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [准备](#准备)
	- [执行环境](#执行环境)
	- [OCT数据](#oct数据)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [参数调整](#参数调整)
	- [patch_shape](#patchshape)
		- [train_ori.py](#trainoripy)
		- [evaluation_ori.py](#evaluationoripy)
	- [学习率与网络结构](#学习率与网络结构)
	- [对某种病理的分类](#对某种病理的分类)

<!-- /TOC -->

本文将介绍基于3D Unet网络而改写的分类网络，其目的是实现对3D OCT体积的病理分类。这里我主要说明其训练与评估的方法。

# 准备

执行代码前的准备主要有两方面，一个是程序相关环境的准备，一个是OCT数据及标签的准备。

## 执行环境

本文的程序是基于3DUnet网络修改而来，所以需要与其源代码相同的执行环境。[3D U-Net Convolution Neural Network with Keras](https://github.com/ellisdg/3DUnetCNN)
```
 pip install tensorflow==1.15.0
```
其他的库与3DUnet的要求一样，可在函数调用时安装缺少的库。

## OCT数据

首先使用 leto-atreides-2/octip 程序对数据进行预处理.将OCT数据保存在每个文件的npy文件中。
然后执行get_label文件获取每组数据的标签。
```
max_list=[]
      for clones in range(15,22):#对列循环
        #if not ('0.0' in data_arr[1:,clones] or '1.0' in data_arr[1:,clones] ):
          #continue
        m=int(float(max(data_arr[1:,clones])))#这里是个坑，我们读出的数据是字符串，要先转化为float型再转化为int型
        max_list.append(m)
      max_list=max(max_list)
      print(max_list)
      if max_list==2 :
        print('=============================================')
        print('attention error')
        max_list = 1
        error=error+1
        print('max_list =')
        print(max_list)
        print('=============================================')
      np.save('label_array',max_list)
```
对于标签的提取，这里for clones in range(15,22)每一列对应一种病理。如果我们想只保存某个病理的标签，我们只需提取对应列的数据。比如clones = 17对应DMLAA。

# 模型训练
1. root_path = '/data_GPU/yihao/3dunet/final/OCT_Images' (99行)

   改为你图片所在的路径。

2. 在当前目录下创建一个空文件夹用来保存模型，比如文件夹original。在开始训练前跳转到这个目录下。（130行）

3. 运行train_ori.py

4. 每个epoch的loss/acc曲线图像保存在logs文件中。通过Tensorboard查看。

5. 选出表现最好的模型（checkpoint文件）。将其从model文件中复制到工作路径（执行训练程序的路径）下。


# 模型评估
1. root_path = '/data_GPU/yihao/3dunet/final/OCT_Images' （52行）

  改为你图片所在的路径。
2. os.chdir("/data_GPU/yihao/3dunet/final")（138行）

   改为你的工作路径（执行训练程序的路径）。
3. os.chdir("/data_GPU/yihao/3dunet/final")（172行）

   改为你保存最佳模型的路径。  
4. model_best=load_model('orimodel-17.h5')（174行）

   改为你最佳模型的名称。
5. 运行evaluation_ori.py。产生的ROC图像保存为当前路径下的roc_ori.png。

# 参数调整
如果你想测试其他参数的表现，你需要改动以下代码。

## patch_shape

### train_ori.py
1. patch_shape=(8, 128, 256)(123行)
2. model1 = unet_model_3d(input_shape=(1, 8, 128, 256)。。。（126行）

### evaluation_ori.py
1. patch_shape=(8, 128, 256)(145行)
2. 因为我们是按体积进行预测，所以要把一个体积中的所欲预测结果取最大值作为整个体积的预测结果。
```
for num in range(66):
   number=num*12
   label_current=max(y_arr[number:number+12])
   pred_current=max(pred_list[number:number+12])
   volume_label.append(label_current)
   volume_predic.append(pred_current)
```
这里的12为一个体积中有12个patch，如果你改动了patch的大小，这个值也需根据patch的大小而变化。

## 学习率与网络结构
学习率与网络结构的改变需在unet3d/model文件夹中对unet.py进行改动。你可以创建不同的unet_model_3d函数并调用。

## 对某种病理的分类
这里用DMLAA举例
首先，你需要获取DMLAA的label，更改并重新执行get_label文件（label_DMLAA.npy）。
在unet3d文件中创建dataDMLAA.py文件，来读取label_DMLAA.npy的信息。
在训练时调用dataDMLAA中的函数。
