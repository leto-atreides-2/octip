import os
import xlrd


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
    g = 'octip_dat'
    for index,nums in enumerate(a):
      if not d in nums:
        if not e in nums:
          if not f in nums:
            if not g in nums:
              if b in nums:
                my_list.append(nums)
              elif c in nums:
                my_list.append(nums)
    return my_list



def main():
    #根目录路径 Chemin du répertoire racine
    root_path = '/data_GPU/yihao/3dunet/final/OCT_Images'
    #用来存放所有的文件路径
    file_list = []
    #用来存放所有的目录路径 Utilisé pour stocker tous les chemins de répertoire
    dir_list = []
    get_file_path(root_path,file_list,dir_list)
    print(dir_list)

    a = dir_list
    b = 'OD'
    c = 'OG'
    list_ODOG=filtre_ODOG(a,b,c)
    print(list_ODOG)

    n = 0
    error=0
    list_csans=[]
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
      #print("数据路径：")#Chemin de données
      print(path_current)
      #///////////////////////////////////////////////
      '''
      提取xlsx文件。
      Extrayez le fichier xml.
      '''
      #用来存放所有的文件路径
      file_list_current = []
      #用来存放所有的目录路径
      dir_list_current = []
      get_file_path(path_current,file_list_current,dir_list_current)
      #print(file_list_current)
      a = file_list_current
      b = 'xlsx'
      xlsx_list = list()
      for index,nums in enumerate(a):
        if b in nums:
          xlsx_list.append(nums)
      print("xlsx文件名称:")#chemin du fichier xlsx
      print(xlsx_list[0])
      xlsx_list = xlsx_list[0]
      #////////////////////////////////////////////////
      '''
      转换为list
      '''
      xlsx_data=xlrd.open_workbook(xlsx_list)  ##获取文本对象
      xlsx_table=xlsx_data.sheets()[0]     ###根据index获取某个sheet
      xlsx_rows=xlsx_table.nrows   ##3获取当前sheet页面的总行数,把每一行数据作为list放到 list
      result=[]
      for i in range(xlsx_rows):
        col=xlsx_table.row_values(i)  ##获取每一列数据
        result.append(col)
      #print(result)
      import numpy as np
      data_arr=np.array(result)
      #print('数据大小')
      #print(data_arr.shape)
      #///////////////////////////////////////////////
      '''
      获取15-21列的病的label
      '''
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


if __name__ == "__main__":
    main()
