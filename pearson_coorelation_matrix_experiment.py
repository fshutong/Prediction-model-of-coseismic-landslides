# coding: utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

df  = pd.read_csv(r'D:\research paper\paper\数据\VIF.CSV')
df.info(); df.head()

lm = ols('landslide~altitude + slope + aspect + curvature + PGA + PGV + fault + river + lith + land', data=df).fit()
lm.summary()

def heatmap(data, method='pearson', camp='Blues', figsize=(10 ,8)):
    """
    data: 整份数据
    method：默认为 pearson 系数
    camp：默认为：RdYlGn-红黄蓝；YlGnBu-黄绿蓝；Blues/Greens 也是不错的选择
    figsize: 默认为 10，8
    """
    # ## 消除斜对角颜色重复的色块

    font1 = {'family': 'Arial', 'size': 10}
    font2 = {'family': 'Arial', 'size': 8}

    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True

    matplotlib.rcParams['agg.path.chunksize'] = 60000
    matplotlib.rcParams.update(matplotlib.rc_params())
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = '10'
    plt.rcParams['figure.figsize'] = (10, 10)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xticks(font=font2)
    # plt.yticks(font=font2)

    plt.figure(figsize=figsize, dpi= 80)
    sns.heatmap(data.corr(method=method), xticklabels=data.corr(method=method).columns, yticklabels=data.corr(method=method).columns, cmap=camp, center=0, annot=True,
                mask = mask)

    plt.savefig(r'D:\research paper\paper\论文\DL实验记录\统计图\线性图.jpeg', dpi=1000)


    plt.show()

heatmap(data=df, figsize=(8,6))



