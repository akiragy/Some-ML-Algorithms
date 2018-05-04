# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from numpy import mat

class NCE(object):
    
    def __init__(self, D=5, Td=500, Tn=1000, n_exp = 20):
        """
        :param D: 数据维度
        :param Td: 待估计样本数量
        :param Tn: 噪声样本数量
        :param n_exp: 重复实验次数，减少随机因素的影响
        """
        self.D, self.Td, self.Tn = D, Td, Tn
        self.n_exp = n_exp
        
        self.max_iter = 1000  #梯度下降最大迭代次数
        self.mu = 0.001  #梯度下降步长
        
        self.list_A_true, self.list_c_true = [], []  #样本分布的真实参数
        self.list_A_opt, self.list_c_opt = [], []  #NCE估计出的样本分布参数
        self.norm_A, self.diff_c = [], []  #参数估计的误差
        

    def create_dataset(self):
        """生成待估计参数的数据集"""
        lmd = 0.8 * np.random.rand(self.D) + 0.1  #特征值在[0.1, 0.9]之间
        M = np.mat(np.random.randn(self.D,self.D))
        U = np.linalg.qr(M)[0]  #QR分解产生正交矩阵
        A_true = U * np.diag(lmd) * U.T  #真实的精度矩阵
        sigma_true = A_true.I  #真实的协方差矩阵
        c_true = -0.5 * np.log(np.linalg.det(sigma_true)) - self.D/2 * np.log(2 * np.pi)  #真实的标准化系数
            
        X = np.random.multivariate_normal([0 for i in range(self.D)], sigma_true, self.Td)  #待估计数据集
        Y = np.random.multivariate_normal([0 for i in range(self.D)], np.eye(self.D), self.Tn)  #噪声数据集，使用标准正态分布
        
        return A_true, c_true, mat(X), mat(Y)
    
    
    def h(self, X, D, A, c):
        g_X = -1/2 * np.diag(X*(A-np.eye(D))*X.T) + c + D/2 * np.log(2 * np.pi)
        h_X = 1 / (1 + np.exp(-g_X))
        return h_X
  
    
    def create_ini(self):
        """迭代的初值"""
        lmd = 10 * np.random.rand(self.D)
        M = np.mat(np.random.randn(self.D,self.D))
        U = np.linalg.qr(M)[0]
        A_ini = U * np.diag(lmd) * U.T
        c_ini = 1
        return A_ini, c_ini
    
    
    def calc_grad(self, X, Y, A, c):
        """计算目标函数对参数A和c的导数"""
        h_X, h_Y = self.h(X, self.D, A, c), self.h(Y, self.D, A, c)
        X_sigma = np.zeros([self.D**2, self.Td])
        for n in range(self.Td):
            X_sigma[:,n] = (X[n,:].T * X[n,:]).reshape(self.D**2)
        Y_sigma = np.zeros([self.D**2, self.Tn])
        for n in range(self.Tn):
            Y_sigma[:,n] = (Y[n,:].T * Y[n,:]).reshape(self.D**2)
            
        df_A = -0.5 * (mat(X_sigma) * np.diag(1-h_X)).sum(axis=1) + 0.5 * (mat(Y_sigma) * np.diag(h_Y)).sum(axis=1)
        df_c = sum(1-h_X) - sum(h_Y)
        df_A = df_A.reshape(self.D,self.D)
        
        return df_A, df_c
    
        
    def gradient_dsecent(self, X, Y):
        """用梯度下降法求解NCE目标函数"""
        A, c = self.create_ini()  #赋初值
        for n_iter in range(self.max_iter):
            df_A, df_c= self.calc_grad(X, Y, A, c)
                   
            A = A + self.mu * df_A
            c = c + self.mu * df_c
            if abs(df_c) < 2e-6:  #终止条件
                break      
            
        return A, c, n_iter
    
    def run_nce(self):

        for i_exp in range(self.n_exp):                  
            A_true, c_true, X, Y = self.create_dataset()
            A_opt, c_opt, n_iter = self.gradient_dsecent(X, Y)
            print("第" + str(i_exp+1) + "次实验的迭代次数为" + str(n_iter))
            print("A的误差：")
            print(np.linalg.norm(A_true-A_opt, 'fro'))
            print("c的误差")
            print(abs(c_opt- c_true))
            
            self.list_A_true.append(A_true)
            self.list_A_opt.append(A_opt)
            self.list_c_true.append(c_true)
            self.list_c_opt.append(c_opt)
            self.norm_A.append(np.linalg.norm(A_true-A_opt, 'fro'))
            self.diff_c.append(abs(c_opt- c_true))
              
nce = NCE(n_exp = 2)
nce.run_nce()
res_A = nce.norm_A
res_c = nce.diff_c