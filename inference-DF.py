import numpy as np
from osgeo import gdal
import pickle
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess=tf.compat.v1.Session(config=config)

# 读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset


# 保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    global im_width, im_height, im_bands
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

model_path =  r"C:\Users\24753\Desktop\DL实验记录\论文图\玩九寨沟.pickle"
Landset_Path = r"C:\Users\24753\Desktop\data\jzg\九寨沟.tif"
SavePath = r"C:\Users\24753\Desktop\landslide\regression\image\结果\玩九寨沟.tif"

dataset = readTif(Landset_Path)
Tif_width = dataset.RasterXSize  # 栅格矩阵的列数
Tif_height = dataset.RasterYSize  # 栅格矩阵的行数e
Tif_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息e
Tif_proj = dataset.GetProjection()  # 获取投影信息
Landset_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)

print(Landset_data.shape[0], Landset_data.shape[1] * Landset_data.shape[2])
################################################调用保存好的模型
# # 以读二进制的方式打开文件
file = open(model_path, "rb")
# 把模型从文件中读取出来
model = pickle.load(file)
# 关闭文件
file.close()

#  在与测试前要调整一下数据的格式
data = np.zeros((Landset_data.shape[0], Landset_data.shape[1] * Landset_data.shape[2]))

for i in range(Landset_data.shape[0]):
    data[i] = Landset_data[i].flatten()
data = data.swapaxes(0, 1)

print(data)
print(data.shape)
#  对调整好格式的数据进行预测
pred = model.predict_proba(data)
pred = pred[:,1]

#  同样地，我们对预测好的数据调整为我们图像的格式
pred = pred.reshape(Landset_data.shape[1], Landset_data.shape[2]) * 255

pred = pred.astype(np.uint8)

#  将结果写到tif图像里
pred = pred.astype(np.uint8)
writeTiff(pred, Tif_geotrans, Tif_proj, SavePath)