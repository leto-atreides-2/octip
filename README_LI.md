OCT图像预处理程序  
Programme de prétraitement d'images OCT
=============
这个代码使用老师的octip程序进行所有图像文件的预处理  
Ce code utilise le programme octip de l'enseignant pour prétraiter tous les fichiers image  
这个程序需要一次执行完毕  
因为每一次都是从头开始执行  
Ce programme doit être fini par seul une fois  
Parce que chaque fois qu'il est exécuté à partir de zéro  
(注意：octip库文件要和图像文件夹在一个路径下）  
(Remarque: le fichier de bibliothèque octip doit être dans le même chemin que le dossier d'image)

# 1.寻找图片路径 Trouver le chemin de l'image
在文件夹中找到所有需要处理的子文件夹的路径。  
Recherchez le chemin de tous les sous-dossiers qui doivent être traités dans le dossier.  
这里认为路径中含OD/OG的子文件夹为所需处理的子文件夹。  
Ici, le sous-dossier avec OD / OG dans le chemin d'accès est considéré comme le sous-dossier à traiter.  
根据自己的路径改root_path  
Changez root_path selon votre propre chemin  

---
首先找到根路径下的所有子路径  
Trouvez d'abord tous les sous-chemins sous le chemin racine
```python
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
 
if __name__ == "__main__":
    #根目录路径 Chemin du répertoire racine
    root_path = '/content/gdrive/My Drive/liyihao/octip-master'
    #用来存放所有的文件路径
    file_list = []
    #用来存放所有的目录路径 Utilisé pour stocker tous les chemins de répertoire
    dir_list = []
    get_file_path(root_path,file_list,dir_list)
    print(dir_list)
```
在所有路径里面寻找带有OG或OD的路径  
注意这里需要排除那些带OG/或OD/的路径  
因为在执行过一次后会在OG或OD文件夹中产生processed和segementation的文件  
需要把这些文件排除  
Trouver tous les chemins avec OG ou OD  
Notez que vous devez exclure ces chemins avec OG / ou OD /  
Parce qu'après une exécution, le fichier traité sera généré dans le dossier OG ou OD  
Ces fichiers doivent être exclus  
```python
a = dir_list
b = 'OD'
c = 'OG'
def filtre_ODOG(a,b,c):
    my_list = list()
    d = 'OD/'
    e = 'OG/'
    for index,nums in enumerate(a):
      if not d in nums:
        if not e in nums:
           if b in nums:
             my_list.append(nums)
           elif c in nums:
               my_list.append(nums)       
    return my_list
list_ODOG=filtre_ODOG(a,b,c)   
print(list_ODOG) 
```
# 2.遍历并执行 Traverser et exécuter
for循环遍历这些文件夹。  
Utilisez une boucle for pour parcourir ces dossiers.  
在每一次遍历的过程中执行图像预处理程序  
Effectuer des procédures de prétraitement d'image lors de chaque traversée  
```python
import octip
import tensorflow as tf
from keras import backend as K
n = 0
for index,nums in enumerate(list_ODOG):
  n = n + 1
  print("第 %d 组数据" %n )# n ième exécution
  #////////////////////////////////////////////////
  '''
  转到要处理的文件夹下。（path_current）
  Accédez au dossier à traiter.（path_current）
  '''
  path_current = nums
  os.chdir(path_current)
  print("数据路径：")#Chemin de données
  print(path_current)
  #///////////////////////////////////////////////
  '''
  提取xml文件。
  Extrayez le fichier xml.
  '''
  #用来存放所有的文件路径
  file_list_current = []
  #用来存放所有的目录路径
  dir_list_current = []
  get_file_path(path_current,file_list_current,dir_list_current)
  #print(file_list_current)
  a = file_list_current
  b = 'xml'
  xml_list = list()
  for index,nums in enumerate(a):
    if b in nums:
      xml_list.append(nums)
  print("xml文件名称")#chemin du fichier xml  
  print(xml_list[0]) 
  path_xml = xml_list[0]
  #////////////////////////////////////////////////
  '''
  开始预处理。
  Démarrez le prétraitement.
  '''
  path_model=root_path+'/octip_models'
  path_segmentation=['octip_data/segmentations1_OCTmain','octip_data/segmentations2_OCTmain']
  path_output='octip_data/preprocessed_OCTmain'

  model_directory = path_model
  segmentation_directories = path_segmentation
  output_directory = path_output

  # parsing the XML file
  bscans = octip.XMLParser(path_xml, False).sorted_bscans()

  # 转换bsacns的输出  Convertir la sortie bsacns
  names_list = []
  for i in range(len(bscans)):
    s = str(i+1)
    new_name = str(path_current) + '/' + s + '.bmp'
    names_list.append(new_name)
  bscans=names_list
  #print(bscans)

  # segmenting the retina in all B-scans with the first model 
  localizer1 = octip.RetinaLocalizer('FPN','efficientnetb6',(384, 384),model_directory = model_directory)
  localizer1(octip.RetinaLocalizationDataset(bscans, 8, localizer1),segmentation_directories[0])

  # segmenting the retina in all B-scans with the second model
  localizer2 = octip.RetinaLocalizer('FPN', 'efficientnetb7', (320, 320),
                                    model_directory = model_directory)
  localizer2(octip.RetinaLocalizationDataset(bscans, 8, localizer2),
            segmentation_directories[1])

  # pre-processing the B-scans
  preprocessor = octip.PreProcessor(200, min_height = 100, normalize_intensities = True)
  preprocessor(bscans, segmentation_directories, output_directory)

  # forming the C-scan
  cscan = octip.bscans_to_cscan(bscans, output_directory, '.png')
  print("cscan.shape = ",cscan.shape)
  np.save('cscan_array',cscan)#把cscan的结果保存在npy文件中，也可以保存为其他形式，例如nii
  K.clear_session()#为了解决OOM问题

  print("当前这组图片处理完毕")#L'ensemble d'images actuel est traité
  ```
产生的预处理文件会在每组图像文件的octip_data文件，cscan结果会保存在cscan_array.npy文件中  
Les fichiers prétraités générés seront dans le fichier octip_data de chaque groupe de fichiers images


