import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, f1_score, precision_score
from sklearn import model_selection
import pickle
from deepforest import CascadeForestClassifier
np.random.seed(6)

def Iris_label(s):
    it={b'no':0, b'yes':1}
    return it[s]

#训练集与测试集
path=r"../train_data.CSV"
#验证集
valpath = r"../test_data.CSV"
#训练模型保存路径
SavePath = r"../model.pickle"

#读取数据集（训练集与测试集）
data=np.loadtxt(path, dtype=float, delimiter=',', converters={10:Iris_label} )
#  converters={n:Iris_label}中“n”指的是第n列：将第3列的str转化为label(number)
#读取数据集（验证集）
valdata = np.loadtxt(valpath, dtype=float, delimiter=',', converters={10:Iris_label} )
#划分数据与标签（训练集与测试集）
x,y=np.split(data,indices_or_sections=(10,),axis=1) #x为数据，y为标签
x=x[:,0:10] #选取前n个波段作为特征
train_data,test_data,train_label,test_label = model_selection.train_test_split(x, y, random_state=1, train_size=0.99,test_size=0.01)
#划分数据与标签（验证集）
valx, valy = np.split(valdata, indices_or_sections=(10,), axis=1) #valx为数据，valy为标签
valx = valx[:,0:10] #选取前n个波段作为特征


train_data = train_data.astype(np.uint8)
test_data = test_data.astype(np.uint8)
train_label = train_label.astype(np.uint8)
test_label = test_label.astype(np.uint8)
#验证集
valx = valx.astype(np.uint8)
valy = valy.astype(np.uint8)

classification = CascadeForestClassifier(max_depth=5,n_trees=100,n_jobs=6, use_predictor="forest")

classification.fit(train_data, train_label.ravel())
prediction = classification.predict_proba(valx)
probability = [prob[1] for prob in prediction]
pre_class = []
for i in probability:
    if i > 0.5:
        pre_class.append(1)
    else:
        pre_class.append(0)
prediction = prediction[:,1]

pred = classification.predict(valx)

print(pred.shape)

print("训练集：", classification.score(train_data, train_label))
print("测试集：", classification.score(test_data, test_label))

a = classification.score(valx, valy)
p = precision_score(valy, pred)
acc = roc_auc_score(valy, pre_class)
roc = roc_auc_score(valy, prediction)
f1_score = f1_score(valy, pred)
cm = confusion_matrix(valy, pred)
recall_score = recall_score(valy, pred)


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


miou = mean_iou(cm)
print('==miou:', miou)


print(a, p, acc, roc, f1_score, cm, recall_score)
#绘制roc图
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve

fpr, tpr, thersholds = roc_curve(valy, prediction, pos_label=None)#非二进制需要pos_label
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label='ROC curve (area = %0.3f)' % roc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
matplotlib.rcParams['agg.path.chunksize'] = 60000
matplotlib.rcParams.update(matplotlib.rc_params())
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(r'..\ROC.jpeg', dpi=3000)
plt.show()

###保存模型
#以二进制的方式打开文件：
file = open(SavePath, "wb")
#将模型写入文件：
pickle.dump(classification, file)
#最后关闭文件：
file.close()


