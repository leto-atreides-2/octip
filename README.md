# OCTIP: traine

dataset path :

```bash

cd /data_GPU/wenpei/script/octip/EtudeOCTBrest/

```
names and labels of datas are saved in dataset_label.txt


train set list: ``` bash  /data_GPU/wenpei/script/octip/train_label.txt```

valid set list: ``` bash  /data_GPU/wenpei/script/octip/valid_label.txt```

test set  list: ``` bash  /data_GPU/wenpei/script/octip/test_label.txt```





to train AHnet 3d without pretraining  

```bash

sbatch 3.sh

```

to train AHnet 3d with pretraining  

```bash

sbatch 5.sh

```

to evaluate AHnet 3d   

```bash

sbatch test_oct.sh

```


to train med 3d   

```bash

sbatch 4.sh

```
to evaluate med3d   

```bash

sbatch test_med3d.sh

```
