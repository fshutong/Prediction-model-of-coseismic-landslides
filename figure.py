import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, f1_score
from tensorflow.python.keras.utils import np_utils
from sklearn import model_selection
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess=tf.compat.v1.Session(config=config)

def Iris_label(s):
    it={b'222':0, b'111':1}
    return it[s]

def readData(filePath):
    data = np.loadtxt(filePath, dtype=float, delimiter=',', converters={10:Iris_label})
    #  converters={3:Iris_label}中“3”指的是第3列：将第3列的str转化为label(number)
    x,y_1D=np.split(data,indices_or_sections=(10,),axis=1) #x为数据，y为标签
    x = x[:, 0:10]  # 选取前X个波段作为特征
    x = np.expand_dims(x, axis=2)
    y = np_utils.to_categorical(y_1D,2)
    # train_data, test_data, train_label, test_label = model_selection.train_test_split(x, y, random_state=1, train_size=0.9, test_size=0.1)
    return x, y, y_1D


#验证集
valpath = r"D:\research paper\paper\数据\总test.CSV"

val_x, val_y, val_y_1D = readData(r'D:\research paper\paper\数据\总test.CSV')



#读取数据集（验证集）
valdata = np.loadtxt(valpath, dtype=float, delimiter=',', converters={10:Iris_label} )

#划分数据与标签（验证集）
valx, valy = np.split(valdata, indices_or_sections=(10,), axis=1) #valx为数据，valy为标签
valx = valx[:,0:10] #选取前n个波段作为特征


#验证集
valx = valx.astype(np.uint8)
valy = valy.astype(np.uint8)


model_path_1 =  r"D:\research paper\paper\数据\DF.pickle"
file_1 = open(model_path_1, "rb")
model_1 = pickle.load(file_1)
file_1.close()
prediction1 = model_1.predict_proba(valx)
prediction1 = prediction1[:,1]
roc1 = roc_auc_score(valy, prediction1)

model_path_2 =  r"D:\research paper\paper\数据\RF.pickle"
file_2 = open(model_path_2, "rb")
model_2 = pickle.load(file_2)
file_2.close()
prediction2 = model_2.predict_proba(valx)
prediction2 = prediction2[:,1]
roc2 = roc_auc_score(valy, prediction2)

model_path_3 =  r"D:\research paper\paper\数据\DT.pickle"
file_3 = open(model_path_3, "rb")
model_3 = pickle.load(file_3)
file_3.close()
prediction3 = model_3.predict_proba(valx)
prediction3 = prediction3[:,1]
roc3 = roc_auc_score(valy, prediction3)

model_path_4 =  r"D:\research paper\paper\数据\MLP.pickle"
file_4 = open(model_path_4, "rb")
model_4 = pickle.load(file_4)
file_4.close()
prediction4 = model_4.predict_proba(valx)
prediction4 = prediction4[:,1]
roc4 = roc_auc_score(valy, prediction4)

model_path_5 = r'D:\research paper\paper\数据\0727-1CNN.h5'
model_5 = load_model(model_path_5)
prediction5 = model_5.predict(val_x)
prediction5 = prediction5[:,1]
roc5 = roc_auc_score(val_y_1D, prediction5)

print(roc1, roc2, roc3, roc4, roc5)







#绘制roc图
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve
from matplotlib.font_manager import FontProperties

# font = FontProperties(fname=r'C:\Windows\Fonts\Arial.ttf')

fpr1, tpr1, thersholds = roc_curve(valy, prediction1, pos_label=None)#非二进制需要pos_label
fpr2, tpr2, thersholds = roc_curve(valy, prediction2, pos_label=None)#非二进制需要pos_label
fpr3, tpr3, thersholds = roc_curve(valy, prediction3, pos_label=None)#非二进制需要pos_label
fpr4, tpr4, thersholds = roc_curve(valy, prediction4, pos_label=None)#非二进制需要pos_label
fpr5, tpr5, thersholds = roc_curve(val_y_1D, prediction5, pos_label=None)#非二进制需要pos_label

lw = 1
plt.plot(fpr1, tpr1, color = 'red', lw = 1, label='DeepForest (area = %0.3f)' % roc1)

plt.plot(fpr2, tpr2, color = 'darkorange', lw = 1, label='RandomForest (area = %0.3f)' % roc2)

plt.plot(fpr3, tpr3, color = 'lime', lw = 1, label='ExtraTrees (area = %0.3f)' % roc3)

plt.plot(fpr4, tpr4, color = 'black', lw = 1, label='MLP (area = %0.3f)' % roc4)

plt.plot(fpr5, tpr5, color = 'blue', lw = 1, label='CNN (area = %0.3f)' % roc5)

plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

font1 = {'family': 'Arial', 'size':10 }
font2 = {'family': 'Arial', 'size':8 }
# font2 = { 'size':7 }

matplotlib.rcParams['agg.path.chunksize'] = 60000
matplotlib.rcParams.update(matplotlib.rc_params())
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = '8'
plt.rcParams['figure.figsize'] = (7.5,7.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(font = font2)
plt.yticks(font = font2)
plt.xlabel('False Positive Rate', font = font1)
plt.ylabel('True Positive Rate', font = font1)
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig(r'D:\research paper\paper\论文\DL实验记录\统计图\测试图.jpeg', dpi=3000)
plt.show()

# import matplotlib.pyplot as plt
# import matplotlib
# from sklearn.metrics import plot_confusion_matrix
# plot_confusion_matrix(classification, valy, pred)


# ###保存模型
# #以二进制的方式打开文件：
# file = open(SavePath, "wb")
# #将模型写入文件：
# # pickle.dump(classifier, file)
# pickle.dump(classification, file)
# #最后关闭文件：
# file.close()
