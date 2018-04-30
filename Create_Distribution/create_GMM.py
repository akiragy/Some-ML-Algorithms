# -- coding: UTF-8 --
"""生成符合混合高斯分布的数据集"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GMM(object):

    def __init__(self, K = 3, d = 3, N = 3000, interval = 3):
        """
        :param K: 高斯成分的数量
        :param d: 高斯分布的维度
        :param N: 要生成的总样本数量
        :param interval: 各个高斯分布均值之间的间隔
        """
        self.K, self.d, self.N = K, d, N
        self.interval = interval

        self.X = np.zeros([N, d])  #样本集
        self.k_true = []  #样本所属成分

        self.mean_true = []  #每个成分的均值
        self.cov_true = []  #每个成分的协方差矩阵


    def create_cov(self):
        """生成协方差矩阵"""
        lmd = 0.8 * np.random.rand(self.d) + 0.1  # 精度矩阵的特征值
        M = np.mat(np.random.randn(self.d, self.d))
        U = np.linalg.qr(M)[0]  # QR分解产生正交矩阵
        A = U * np.diag(lmd) * U.T  # 精度矩阵
        return A.I  #协方差矩阵


    def select_class(self):
        """根据混合系数选择一个高斯成分"""
        c_ran = np.random.rand(1)
        sum_k = 0
        for i, k in enumerate(self.mix_coef):
            sum_k += k
            if c_ran <= sum_k:
                return i
        return -1


    def sample_from_gauss(self, k_sltd):
        """从单个高斯成分中采样"""
        assert k_sltd != -1
        mean_sltd, cov_sltd = self.mean_true[k_sltd], self.cov_true[k_sltd]  #选择均值和协方差矩阵
        x = np.random.multivariate_normal(mean_sltd, cov_sltd, 1)
        return x


    def create_dataset(self):
        """生成符合GMM的数据集"""
        for c_iter in range(self.K):
            cur_mean = self.interval * ( np.random.rand(self.d) + 3 * c_iter)  #当前高斯成分的均值

            cur_cov = self.create_cov()  #当前高斯成分的协方差矩阵
            self.mean_true.append(cur_mean)
            self.cov_true.append(cur_cov)

        self.mix_coef = np.random.rand(self.K) + 0.3  #为了防止有的混合系数过小，故而加个数软化一下
        self.mix_coef = self.mix_coef / sum(self.mix_coef)  #混合系数

        for n_iter in range(self.N):
            k_sltd = self.select_class()  #根据混合系数选择一个高斯成分
            self.k_true.append(k_sltd)
            self.X[n_iter,:] = self.sample_from_gauss(k_sltd)  #从所选高斯成分中采样


    def plt_hist(self):
        """样本直方图"""
        plt.hist(self.X, normed=True, bins=100)  #将每个维度上的直方图显示在一起
        plt.title("histogram of GMM dataset")
        plt.show()


    def plt_3d(self, d0=0, d1=1, d2=2):
        """样本3维散点图"""
        if self.d < 3:
            print("样本维度不足3，无法画出3D散点图")
            return
        if max(d0,d1,d2) + 1 > self.d:
            print("维度越界")
            return

        ax = plt.subplot(111, projection='3d')  #需要导入Axes3D
        for k_iter in range(self.K):
            k_n = [i for i in range(self.N) if self.k_true[i] == k_iter]
            ax.scatter(self.X[k_n,d0], self.X[k_n,d1], self.X[k_n,d2])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        ax.legend(('group1', 'group2', 'group3'))
        ax.set_title('scatter diagram of GMM dataset', loc='center', fontsize='large')
        plt.show()


if __name__ == "__main__":
    gmm = GMM(d = 4)
    gmm.create_dataset()
    gmm.plt_hist()
    gmm.plt_3d()