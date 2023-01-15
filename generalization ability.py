import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, f1_score, precision_score, cohen_kappa_score
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
valpath = r"../data.CSV"

val_x, val_y, val_y_1D = readData(r'../data.CSV')



#读取数据集（验证集）
valdata = np.loadtxt(valpath, dtype=float, delimiter=',', converters={10:Iris_label} )

#划分数据与标签（验证集）
valx, valy = np.split(valdata, indices_or_sections=(10,), axis=1) #valx为数据，valy为标签
valx = valx[:,0:10] #选取前n个波段作为特征


#验证集
valx = valx.astype(np.uint8)
valy = valy.astype(np.uint8)


model_path_1 =  r"../gcf4.pickle"
file_1 = open(model_path_1, "rb")
model_1 = pickle.load(file_1)
file_1.close()
prediction1 = model_1.predict_proba(valx)
probability1 = [prob[1] for prob in prediction1]
# scio.savemat('', probability)
pre_class1 = []
for i in probability1:
    if i > 0.5:
        pre_class1.append(1)
    else:
        pre_class1.append(0)
prediction1 = prediction1[:,1]
pred1 = model_1.predict(valx)
a1 = model_1.score(valx, valy)
p1 = precision_score(valy, pred1)
acc1 = roc_auc_score(valy, pre_class1)
roc1 = roc_auc_score(valy, prediction1)
f1_score1 = f1_score(valy, pred1)
cm1 = confusion_matrix(valy, pred1)
recall_score1 = recall_score(valy, pred1)
ka1 = cohen_kappa_score(valy, pred1)
#print(a1, p1, acc1, roc1, f1_score1, cm1, recall_score1)


model_path_2 =  r"../gc.pickle"
file_2 = open(model_path_2, "rb")
model_2 = pickle.load(file_2)
file_2.close()
prediction2 = model_2.predict_proba(valx)
probability2 = [prob[1] for prob in prediction2]
# scio.savemat('', probability)
pre_class2 = []
for i in probability2:
    if i > 0.5:
        pre_class2.append(1)
    else:
        pre_class2.append(0)
prediction2 = prediction2[:,1]
pred2 = model_2.predict(valx)
a2 = model_2.score(valx, valy)
p2 = precision_score(valy, pred2)
acc2 = roc_auc_score(valy, pre_class2)
roc2 = roc_auc_score(valy, prediction2)
f1_score2 = f1_score(valy, pred2)
cm2 = confusion_matrix(valy, pred2)
recall_score2 = recall_score(valy, pred2)
ka2 = cohen_kappa_score(valy, pred2)
#print(a2, p2, acc2, roc2, f1_score2, cm2, recall_score2)

model_path_3 =  r"../DT.pickle"
file_3 = open(model_path_3, "rb")
model_3 = pickle.load(file_3)
file_3.close()
prediction3 = model_3.predict_proba(valx)
probability3 = [prob[1] for prob in prediction3]
# scio.savemat('', probability)
pre_class3 = []
for i in probability3:
    if i > 0.5:
        pre_class3.append(1)
    else:
        pre_class3.append(0)
prediction3 = prediction3[:,1]
pred3 = model_3.predict(valx)
a3 = model_3.score(valx, valy)
p3 = precision_score(valy, pred3)
acc3 = roc_auc_score(valy, pre_class3)
roc3 = roc_auc_score(valy, prediction3)
f1_score3 = f1_score(valy, pred3)
cm3 = confusion_matrix(valy, pred3)
recall_score3 = recall_score(valy, pred3)
ka3 = cohen_kappa_score(valy, pred3)
#print(a3, p3, acc3, roc3, f1_score3, cm3, recall_score3)

model_path_4 =  r"../MLP.pickle"
file_4 = open(model_path_4, "rb")
model_4 = pickle.load(file_4)
file_4.close()
prediction4 = model_4.predict_proba(valx)
probability4 = [prob[1] for prob in prediction4]
# scio.savemat('', probability)
pre_class4 = []
for i in probability4:
    if i > 0.5:
        pre_class4.append(1)
    else:
        pre_class4.append(0)
prediction4 = prediction4[:,1]
pred4 = model_4.predict(valx)
a4 = model_4.score(valx, valy)
p4 = precision_score(valy, pred4)
acc4 = roc_auc_score(valy, pre_class4)
roc4 = roc_auc_score(valy, prediction4)
f1_score4 = f1_score(valy, pred4)
cm4 = confusion_matrix(valy, pred4)
recall_score4 = recall_score(valy, pred4)
ka4 = cohen_kappa_score(valy, pred4)
#print(a4, p4, acc4, roc4, f1_score4, cm4, recall_score4)

def compute_accuracy(real_y,predict_y):
    length = len(real_y)
    correct = []
    for i in range(length):
        if real_y[i] == predict_y[i]:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = np.mean(correct) #np.mean对所有值求平均
    return accuracy


model_path_5 = r'../CNN.h5'
model_5 = load_model(model_path_5)
prediction5 = model_5.predict(val_x)
probability5 = [prob[1] for prob in prediction5]
pre_class5 = []
for i in probability5:
    if i > 0.5:
        pre_class5.append(1)
    else:
        pre_class5.append(0)
prediction5 = prediction5[:,1]
# pred5 = model_5.predict(val_x)
a5 = compute_accuracy(val_y_1D, pre_class5)
p5 = precision_score(val_y_1D, pre_class5)
acc5 = roc_auc_score(val_y_1D, pre_class5)
roc5 = roc_auc_score(val_y_1D, prediction5)
f1_score5 = f1_score(val_y_1D, pre_class5)
cm5 = confusion_matrix(val_y_1D, pre_class5)
recall_score5 = recall_score(val_y_1D, pre_class5)
ka5 = cohen_kappa_score(val_y_1D, pre_class5)


def mean_iou(cf_mtx):
    """
    cf_mtx(ndarray): shape -> (class_num, class_num), 混淆矩阵
    """
    #
    mIou = np.diag(cf_mtx) / (np.sum(cf_mtx, axis=1) + \
                              np.sum(cf_mtx, axis=0) - np.diag(cf_mtx))
    print('===mIou:', mIou)
    # 所有类别iou取平均
    mIou = np.nanmean(mIou)
    return mIou

miou1 = mean_iou(cm1)
miou2 = mean_iou(cm2)
miou3 = mean_iou(cm3)
miou4 = mean_iou(cm4)
miou5 = mean_iou(cm5)


print(a1,  roc1, ka1,  f1_score1, miou1)
print(a2,  roc2, ka2,  f1_score2, miou2)
print(a3,  roc3, ka3,  f1_score3, miou3)
print(a4,  roc4, ka4,  f1_score4, miou4)
print(a5,  roc5, ka5,  f1_score5, miou5)






# #绘制roc图
# import matplotlib.pyplot as plt
# import matplotlib
# from sklearn.metrics import roc_curve
# from matplotlib.font_manager import FontProperties
#
# # font = FontProperties(fname=r'C:\Windows\Fonts\Arial.ttf')
#
# fpr1, tpr1, thersholds = roc_curve(valy, prediction1, pos_label=None)#非二进制需要pos_label
# fpr2, tpr2, thersholds = roc_curve(valy, prediction2, pos_label=None)#非二进制需要pos_label
# fpr3, tpr3, thersholds = roc_curve(valy, prediction3, pos_label=None)#非二进制需要pos_label
# fpr4, tpr4, thersholds = roc_curve(valy, prediction4, pos_label=None)#非二进制需要pos_label
# fpr5, tpr5, thersholds = roc_curve(val_y_1D, prediction5, pos_label=None)#非二进制需要pos_label
#
# lw = 1
# plt.plot(fpr1, tpr1, color = 'red', lw = 1, label='DeepForest (area = %0.3f)' % roc1)
#
# plt.plot(fpr2, tpr2, color = 'darkorange', lw = 1, label='RandomForest (area = %0.3f)' % roc2)
#
# plt.plot(fpr3, tpr3, color = 'lime', lw = 1, label='ExtraTrees (area = %0.3f)' % roc3)
#
# plt.plot(fpr4, tpr4, color = 'black', lw = 1, label='MLP (area = %0.3f)' % roc4)
#
# plt.plot(fpr5, tpr5, color = 'blue', lw = 1, label='CNN (area = %0.3f)' % roc5)
#
# plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
#
# font1 = {'family': 'Arial', 'size':10 }
# # font2 = { 'size':7 }
#
# matplotlib.rcParams['agg.path.chunksize'] = 60000
# matplotlib.rcParams.update(matplotlib.rc_params())
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = '8'
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', font1)
# plt.ylabel('True Positive Rate', font1)
# # plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig(r'../roc.jpeg', dpi=3000)
# plt.show()
#
# # import matplotlib.pyplot as plt
# # import matplotlib
# # from sklearn.metrics import plot_confusion_matrix
# # plot_confusion_matrix(classification, valy, pred)
#
#
# # ###保存模型
# # #以二进制的方式打开文件：
# # file = open(SavePath, "wb")
# # #将模型写入文件：
# # # pickle.dump(classifier, file)
# # pickle.dump(classification, file)
# # #最后关闭文件：
# # file.close()
