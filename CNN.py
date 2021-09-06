#!/usr/bin/python
# -*- coding=utf-8 -*-
import os
import numpy as np
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,Flatten,BatchNormalization,Dropout
from sklearn import metrics
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from numpy import random
import tensorflow as tf
config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess=tf.compat.v1.Session(config=config)

tf.random.set_seed(6)
np.random.seed(6)


#  定义字典，便于来解析样本数据集txt
def Iris_label(s):
    it={b'222':0, b'111':1}
    return it[s]

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        # if loss_type == 'epoch':
            # val_acc
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

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


def readData(filePath):
    data = np.loadtxt(filePath, dtype=float, delimiter=',', converters={10:Iris_label})
    x,y_1D=np.split(data,indices_or_sections=(10,),axis=1) #x为数据，y为标签
    x = x[:, 0:10]  # 选取前X个波段作为特征
    x = np.expand_dims(x, axis=2)
    y = np_utils.to_categorical(y_1D,2)
    return x, y, y_1D

train_x, train_y , train_y_1D= readData(r'C:\Users\24753\Desktop\data\测试\总\汶川芦山鲁甸米林.CSV')
test_x, test_y , test_y_1D= readData(r'C:\Users\24753\Desktop\data\测试\总\data.CSV')
val_x, val_y, val_y_1D = readData(r'C:\Users\24753\Desktop\data\jzg\data.CSV')

model = Sequential()

model.add(Conv1D(15, 3, activation='tanh', input_shape=(10, 1)))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(30, activation='tanh'))
model.add(Dropout(rate=0.05))
model.add(Dense(15, activation='tanh'))
model.add(Dense(2, activation='softmax'))
optimizer = optimizers.Adagrad(lr=0.006)
model.compile(loss='categorical_crossentropy',
             optimizer=optimizer,metrics=['accuracy'])
print(model.summary())
history = LossHistory()

model.fit(train_x,train_y,validation_data=(test_x,test_y),callbacks=[history],
          verbose=2, epochs=100, batch_size=4800, shuffle=True)

dense1_layer_model = Model(inputs = model.input, outputs = model.layers[-2].output)

print(test_x)
print(test_x.shape)
y_pred_val = model.predict(val_x)
probability = [prob[1] for prob in y_pred_val]
pre_class = []
for i in probability:
    if i > 0.5:
        pre_class.append(1)
    else:
        pre_class.append(0)
accuracy = compute_accuracy(val_y_1D, pre_class)
print(accuracy)
test_auc = metrics.roc_auc_score(val_y_1D, probability)
print(test_auc)
f1_score = metrics.f1_score(val_y_1D, pre_class)
print(f1_score)
cm = metrics.confusion_matrix(val_y_1D, pre_class)
print(cm)
recall_score = metrics.recall_score(val_y_1D, pre_class)
print(recall_score)
mcc = metrics.matthews_corrcoef(val_y_1D, pre_class)
print(mcc)

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

import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve

fpr, tpr, thersholds = roc_curve(val_y_1D, probability, pos_label=None)#非二进制需要pos_label
lw = 2
plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label='ROC curve (area = %0.3f)' % test_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
matplotlib.rcParams['agg.path.chunksize'] = 60000
matplotlib.rcParams.update(matplotlib.rc_params())
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig(r'C:\Users\24753\Desktop\data\AUC曲线\DF\汶川.jpeg', dpi=300)
plt.show()

miou = mean_iou(cm)
print('==miou:', miou)
model.save(r'C:\Users\24753\Desktop\landslide\regression\model\other\0727-1CNN.h5')
dense1_layer_model.save(r'C:\Users\24753\Desktop\landslide\regression\model\other\0727-1CNN-2.h5')
history.loss_plot('epoch')