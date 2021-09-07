from osgeo import gdal
import os
import random


# 读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


Landset_Path = r"C:\Users\24753\Desktop\data\jzg\jzg.tif"
LabelPath = r"C:\Users\24753\Desktop\data\jzg\landslide1.tif"
txt_Path = r"C:\Users\24753\Desktop\data\jzg\valdata.CSV"

##########################################################  读取图像数据
dataset = readTif(Landset_Path)
Tif_width = dataset.RasterXSize  # 栅格矩阵的列数
Tif_height = dataset.RasterYSize  # 栅格矩阵的行数
Tif_bands = dataset.RasterCount  # 波段数
Tif_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
Landset_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)
dataset = readTif(LabelPath)
Tif_width = dataset.RasterXSize  # 栅格矩阵的列数
Tif_height = dataset.RasterYSize  # 栅格矩阵的行数
Tif_bands = dataset.RasterCount  # 波段数
Tif_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
Label_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)


# 写之前，先检验文件是否存在，存在就删掉
if os.path.exists(txt_Path):
    os.remove(txt_Path)
# 以写的方式打开文件，如果文件不存在，就会自动创建
file_write_obj = open(txt_Path, 'w')

####################################################首先收集滑坡样本，
####################################################遍历所有像素值，
####################################################随机收集滑坡样本。
# 随机采样
count = 0
for i in range(50000000):
    X_random = random.randint(0, Label_data.shape[0] - 1)
    Y_random = random.randint(0, Label_data.shape[1] - 1)
    #  设置的滑坡类别在标签图中像元值为1
    if (Label_data[X_random][Y_random] == 1):
        var = ""
        for k in range(Landset_data.shape[0]):
            var = var + str(Landset_data[k][X_random][Y_random]) + ","
        var = var + "yes"
        file_write_obj.writelines(var)
        file_write_obj.write('\n')
        count = count + 1
    if (count ==10000):
        break

####################################################非滑坡样本数量与滑坡样本数量保持一致。

Threshold = count
count = 0
for i in range(8000000):
    X_random = random.randint(0, Label_data.shape[0] - 1)
    Y_random = random.randint(0, Label_data.shape[1] - 1)
    #  设置的非滑坡类别在标签图中像元值为0
    if (Label_data[X_random][Y_random] == 0):
        var = ""
        for k in range(Landset_data.shape[0]):
            var = var + str(Landset_data[k][X_random][Y_random]) + ","
        var = var + "no"
        file_write_obj.writelines(var)
        file_write_obj.write('\n')
        count = count + 1
    if (count == Threshold):
        break

file_write_obj.close()

#全采样
####################################################首先收集滑坡样本，
####################################################遍历所有像素值，
####################################################收集所有滑坡样本。
#
# count = 0
# for i in range(Label_data.shape[0]):
#     for j in range(Label_data.shape[1]):
#         #  设置的滑坡类别在标签图中像元值为1
#         if (Label_data[i][j] == 1):
#             var = ""
#             for k in range(Landset_data.shape[0]):
#                 var = var + str(Landset_data[k][i][j]) + ","
#             var = var + "yes"
#             file_write_obj.writelines(var)
#             file_write_obj.write('\n')
#             count = count + 1
#
# ####################################################非滑坡样本数量与滑坡样本数量保持一致。
#
# Threshold = count
# count = 0
# for i in range(8000000):
#     X_random = random.randint(0, Label_data.shape[0] - 1)
#     Y_random = random.randint(0, Label_data.shape[1] - 1)
#     #  设置的非滑坡类别在标签图中像元值为0
#     if (Label_data[X_random][Y_random] == 0):
#         var = ""
#         for k in range(Landset_data.shape[0]):
#             var = var + str(Landset_data[k][X_random][Y_random]) + ","
#         var = var + "no"
#         file_write_obj.writelines(var)
#         file_write_obj.write('\n')
#         count = count + 1
#     if (count == Threshold):
#         break
#
# file_write_obj.close()