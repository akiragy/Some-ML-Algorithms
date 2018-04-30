# -- coding: UTF-8 --
"""用EM算法对混合高斯分布样本集进行聚类"""
import sys
sys.path.append("..")
from Create_Distribution import create_GMM

import numpy as np
from numpy import mat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EM_for_GMM(object):

    def __init__(self, K = 3, d = 3, N = 3000, max_iter = 200):
        """
        :param K: 聚蔟数量
        :param d: 样本维度
        :param N: 样本数量
        :param max_iter: 最大迭代次数
        """
        np.random.seed(2018)

        self.gmm = create_GMM.GMM(K, d, N, interval=1)  #生成混合高斯分布的数据集
        self.gmm.create_dataset()

        self.K, self.d, self.N = K, d, N
        self.max_iter = max_iter
        self.iter = self.max_iter  #实际迭代次数

        self.X =self.gmm.X  #样本集
        self.k_true = self.gmm.k_true  #样本所属聚蔟

        self.mean_true = self.gmm.mean_true  #每个聚蔟的均值
        self.cov_true = self.gmm.cov_true  #每个聚蔟的协方差矩阵


    def pdf_gauss(self, d, mean, cov, x):
        """d维高斯分布的pdf"""
        c = (2*np.pi)**(-0.5*d) * np.linalg.det(cov)**(-0.5)
        quad = -0.5 * mat(x-mean) * mat(cov).I * mat(x-mean).T
        p = c * np.exp(quad)
        return p


    def E_step(self):
        self.gamma = np.zeros([self.N, self.K])  #后验概率
        for n_iter in range(self.N):
            z = np.zeros(self.K)
            for k in range(self.K):
                z[k] = self.mix_coef_em[k] * self.pdf_gauss(self.d,self.mean_em[k],self.cov_em[k],self.X[n_iter,:])
            z = z / sum(z)
            self.gamma[n_iter,:] = z


    def M_step(self):
        for k_iter in range(self.K):
            mm, mc = 0, 0
            for n_iter in range(self.N):
                mm += self.gamma[n_iter,k_iter] * self.X[n_iter,:]
            self.mean_em[k_iter] = mm / sum(self.gamma[:,k_iter])  #更新均值
            for n_iter in range(self.N):
                mc += self.gamma[n_iter,k_iter] * mat(self.X[n_iter,:]-self.mean_em[k_iter]).T * mat(self.X[n_iter,:]-self.mean_em[k_iter])
            self.cov_em[k_iter] = mc / sum(self.gamma[:,k_iter])  #更新协方差矩阵
            self.mix_coef_em[k_iter] = sum(self.gamma[:,k_iter]) / self.N  #更新混合系数


    def calc_ll(self):
        """计算似然函数"""
        ll = 0
        for n in range(self.N):
            sll = 0
            for k in range(self.K):
                sll += self.mix_coef_em[k] * self.pdf_gauss(self.d,self.mean_em[k],self.cov_em[k],self.X[n,:])
            ll += np.log(sll)
        return ll[0,0]


    def sep_class(self):
        """根据EM算法结果划分聚簇"""
        self.k_em = []
        for n_iter in range(self.N):
            gamma_n = self.gamma[n_iter,:]
            k_n = [i for i in range(self.K) if gamma_n[i] == max(gamma_n)]
            self.k_em.append(k_n[0])


    def EM_alg(self):

        self.mean_em = [self.X[i,:] for i in range(self.K)]  #初始化均值
        self.cov_em = [np.eye(self.d) for i in range(self.K)]  #初始化协方差矩阵
        self.mix_coef_em = [1.0/self.K for i in range(self.K)]  #初始化混合系数

        ll_cur = self.calc_ll()
        for n_iter in range(self.max_iter):
            ll_pre = ll_cur
            self.E_step()
            self.M_step()
            ll_cur = self.calc_ll()

            if n_iter % 10 == 0:
                print("第" + str(n_iter+1) + "次迭代后的似然函数为" + str(ll_cur))

            if ll_cur - ll_pre < 1e-4:
                self.iter = n_iter + 1
                break
        self.sep_class()


    def plt_2d(self, ax_0=0, ax_1=1):
        """作任意两个维度之间的散点图"""
        plt.subplot(121)  #groundtruth
        for k_iter in range(self.K):
            k_n = [i for i in range(self.N) if self.k_true[i] == k_iter]
            plt.scatter(self.X[k_n,ax_0], self.X[k_n,ax_1])

        plt.subplot(122)  #算法结果
        for k_iter in range(self.K):
            k_n = [i for i in range(self.N) if self.k_em[i] == k_iter]
            plt.scatter(self.X[k_n,ax_0], self.X[k_n,ax_1])
        plt.show()


    def plt_3d(self, d0=0, d1=1, d2=2):
        '''作3维散点图'''
        if self.d < 3:
            print("样本维度不足3，无法画出3D散点图")
            return
        if max(d0,d1,d2) + 1 > self.d:
            print("维度越界")
            return

        ax = plt.subplot(121, projection='3d')  #groundtruth
        for k_iter in range(self.K):
            k_n = [i for i in range(self.N) if self.k_true[i] == k_iter]
            ax.scatter(self.X[k_n,d0], self.X[k_n,d1], self.X[k_n,d2])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        ax.legend(('group1', 'group2', 'group3'))
        ax.set_title('groundtruth', loc='center', fontsize='large')

        ax = plt.subplot(122, projection='3d')  #算法结果
        for k_iter in range(self.K):
            k_n = [i for i in range(self.N) if self.k_em[i] == k_iter]
            ax.scatter(self.X[k_n,d0], self.X[k_n,d1], self.X[k_n,d2])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        ax.legend(('group1', 'group2', 'group3'))
        ax.set_title('result of EM', loc='center', fontsize='large')

        plt.show()


if __name__ == "__main__":

    ee = EM_for_GMM()
    ee.EM_alg()
    ee.plt_2d(0,1)
    ee.plt_3d()