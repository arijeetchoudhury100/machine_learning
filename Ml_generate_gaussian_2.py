import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cov1 = [[1,0],[0,1]]
cov2 = [[0.2,0],[0,0.2]]
cov3 = [[2,0],[0,2]]
cov4 = [[0.2,0],[0,2]]
cov5 = [[2,0],[0,0.2]]
cov6 = [[1,0.5],[0.5,1]]
cov7 = [[0.3,0.5],[0.5,2]]
cov8 = [[0.3,-0.5],[-0.5,2]]
mu = [0,0]

fig,axes = plt.subplots(nrows=3,ncols=3)
x1,x2 = np.random.multivariate_normal(mu,cov1,500).T
x3,x4 = np.random.multivariate_normal(mu,cov2,500).T
x5,x6 = np.random.multivariate_normal(mu,cov3,500).T
x7,x8 = np.random.multivariate_normal(mu,cov4,500).T
x9,x10 = np.random.multivariate_normal(mu,cov5,500).T
x11,x12 = np.random.multivariate_normal(mu,cov6,500).T
x13,x14 = np.random.multivariate_normal(mu,cov7,500).T
x15,x16 = np.random.multivariate_normal(mu,cov8,500).T

axes[0][0].scatter(x1,x2)
axes[0][1].scatter(x3,x4)
axes[0][2].scatter(x5,x6)
axes[1][0].scatter(x7,x8)
axes[1][1].scatter(x9,x10)
axes[1][2].scatter(x11,x12)
axes[2][0].scatter(x13,x14)
axes[2][1].scatter(x15,x16)
plt.tight_layout()

axes[0][0].set_title('s1 = 1,s2 = 1,s12 = s21 = 0');
axes[0][1].set_title('s1 = 0.2,s2 = 0.2,s12 = s21 = 0');
axes[0][2].set_title('s1 = 2,s2 = 2,s12 = s21 = 0');
axes[1][0].set_title('s1 = 0.2,s2 = 2,s12 = s21 = 0');
axes[1][1].set_title('s1 = 2,s2 = 0.2,s12 = s21 = 0');
axes[1][2].set_title('s1 = 1,s2 = 1,s12 = s21 = 0.5');
axes[2][0].set_title('s1 = 0.3,s2 = 0.2,s12 = s21 = 0.5');
axes[2][1].set_title('s1 = 0.3,s2 = 2,s12 = s21 = -0.5'); 

plt.show()
