3D UNet for classification of OCT volume
=============================================

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Préparation](#prparation)
	- [Environnement d'exécution](#environnement-dexcution)
	- [Volumes OCT](#volumes-oct)
- [L'entraînement du modèle(train_ori.py)](#lentranement-du-modletrainoripy)
- [Évaluation du modèle(evaluation_ori.py)](#valuation-du-modleevaluationoripy)
- [Paramètre](#paramtre)
	- [patch_shape](#patchshape)
		- [train_ori.py](#trainoripy)
		- [evaluation_ori.py](#evaluationoripy)
	- [Taux d'apprentissage et structure du réseau](#taux-dapprentissage-et-structure-du-rseau)
- [Classification d'une certaine pathologie](#classification-dune-certaine-pathologie)

<!-- /TOC -->

Cet article présente un réseau de classification réécrit à partir du réseau 3D Unet, dont le but est de réaliser une classification pathologique des volumes 3D OCT. J'explique ici principalement ses méthodes de formation et d'évaluation.

# Préparation

Il y a deux aspects principaux de la préparation avant d'exécuter le code, l'un est la préparation de l'environnement lié au programme et l'autre est la préparation des données et des volumes OCT.

## Environnement d'exécution

Le programme de cet article est modifié en fonction du réseau 3DUnet, il a donc besoin du même environnement d'exécution que son code source.[3D U-Net Convolution Neural Network with Keras](https://github.com/ellisdg/3DUnetCNN)
```
 pip install tensorflow==1.15.0
```
D'autres bibliothèques ont les mêmes exigences que 3DUnet, et les bibliothèques manquantes peuvent être installées lorsque la fonction est appelée.

## Volumes OCT

Utilisez d'abord le programme leto-atreides-2 / octip pour prétraiter les données et enregistrez les données OCT dans le fichier npy de chaque fichier.
Exécutez ensuite le fichier get_label.py pour obtenir le libellé de chaque groupe de données.
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
Pour l'extraction des étiquettes, for clones in range(15,22) chaque colonne correspond à une pathologie. Si nous voulons enregistrer uniquement l'étiquette d'une certaine pathologie, il suffit d'extraire les données de la colonne correspondante. Par exemple, clones = 17 correspond à DMLAA.

# L'entraînement du modèle(train_ori.py)
1. root_path = '/data_GPU/yihao/3dunet/final/OCT_Images' (Ligne 99)

  Changez pour le chemin où se trouve votre image.

2. Créez un dossier vide dans le répertoire actuel pour enregistrer le modèle. Accédez à ce répertoire avant de commencer la formation. (Ligne 130)

3. Exécutez train_ori.py

4. L'image de la courbe de loss / acc de chaque époque est enregistrée dans le fichier logs. Vue à travers Tensorboard.

5. Sélectionnez le modèle le plus performant (fichier checkpoint). Copiez-le du fichier de modèle vers le chemin de travail (le chemin où le programme de formation est exécuté).


# Évaluation du modèle(evaluation_ori.py)
1. root_path = '/data_GPU/yihao/3dunet/final/OCT_Images' （Ligne 52）

  Changez pour le chemin où se trouve votre image.
2. os.chdir("/data_GPU/yihao/3dunet/final")（Ligne 138）

   Modifiez votre chemin de travail (le chemin pour exécuter le programme de formation).
3. os.chdir("/data_GPU/yihao/3dunet/final")（Ligne 172）

  Changez pour le chemin où vous avez enregistré le meilleur modèle.
4. model_best=load_model('orimodel-17.h5')（Ligne 174）

   Changez pour le nom de votre meilleur modèle.
5. Exécutez evaluation_ori.py. L'image ROC générée est enregistrée sous roc_ori.png dans le chemin actuel.

# Paramètre
Si vous souhaitez tester les performances d'autres paramètres, vous devez modifier le code suivant.

## patch_shape

### train_ori.py
1. patch_shape=(8, 128, 256)(Ligne 123)
2. model1 = unet_model_3d(input_shape=(1, 8, 128, 256)。。。（Ligne 126）

### evaluation_ori.py
1. patch_shape=(8, 128, 256)(Ligne 145)
2. Parce que nous prédisons en volume, nous devons prendre la valeur maximale de tous les résultats de prédiction dans un volume comme résultat de prédiction du volume entier.
```
for num in range(66):
   number=num*12
   label_current=max(y_arr[number:number+12])
   pred_current=max(pred_list[number:number+12])
   volume_label.append(label_current)
   volume_predic.append(pred_current)
```
Ici, 12 signifie qu'il y a 12 patches dans un volume. Si vous changez la taille du patch, cette valeur doit également changer en fonction de la taille du patch.

## Taux d'apprentissage et structure du réseau
Le taux d'apprentissage et la structure du réseau doivent être modifiés dans unet.py dans le dossier unet3d / model. Vous pouvez créer différentes fonctions unet_model_3d et les appeler.

# Classification d'une certaine pathologie
Voici un exemple avec DMLAA

Tout d'abord, vous devez obtenir l'étiquette de DMLAA, modifier et réexécuter le fichier get_label (label_DMLAA.npy).

Créez le fichier dataDMLAA.py dans le fichier unet3d pour lire les informations label_DMLAA.npy.

Appelez la fonction dans dataDMLAA pendant l'entraînement.
