import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 假设我们有一个数据集，其中包括两个共线性特征 A 和 B
def plot_gaussianmixture(df,cola, colb):
    df = df[[cola, colb]]
    # 分割数据为训练和测试集
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # 使用 GMM 来估计联合分布
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(train_data)

    # 我们可以绘制特征 A 和 B 的联合分布
    x, y = np.meshgrid(np.linspace(df[cola].min(), df[cola].max(), 100),
                    np.linspace(df[colb].min(), df[colb].max(), 100))

    xy = np.array([x.ravel(), y.ravel()]).T

    # 预测联合分布概率密度
    z = np.exp(gmm.score_samples(xy))
    z = z.reshape(x.shape)

    # 绘制联合分布
    plt.contourf(x, y, z, levels=20, cmap='Blues')
    plt.colorbar(label='Probability Density')
    plt.xlabel(cola)
    plt.ylabel(colb)
    plt.title(f'GMM Estimated Joint Distribution of {cola} and {colb}')
    fig = plt.gcf()
    return fig
