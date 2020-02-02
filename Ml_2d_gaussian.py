import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_multivariate(x,cov,mu):
	det_cov = np.linalg.det(cov)
	inv_cov = np.linalg.inv(cov)
	y = ((1/((2*np.pi)*np.sqrt(det_cov))) * np.e**(-1*0.5*(x-mu).dot(inv_cov).dot((x-mu).reshape(2,1))))
	return y[0][0]

cov = np.array([[1,0],[0,1]])
mu = np.zeros((1,2))

X =[]
Y = []
Z = []

for i in range(0,1000):
	x = np.random.randn(1,2)
	X.append(x[0][0])
	Y.append(x[0][1])
	Z.append(gaussian_multivariate(x,cov,mu))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(X,Y,Z)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('probability')
ax.set_title('2D gaussian plot')
plt.show()
	
