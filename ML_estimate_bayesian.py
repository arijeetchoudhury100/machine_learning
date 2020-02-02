import numpy as np
import matplotlib.pyplot as plt

def classify(a,b,cov):
	cov_inv = np.linalg.inv(cov)
	cov_det = np.linalg.det(cov)
	a = np.delete(a,3,1)
	g = -0.5*((a-b).dot(cov_inv).dot((a-b).transpose()))[0][0] -0.5*np.log(cov_det)
	return g

def classify2(a,b,cov,p):
	cov_inv = np.linalg.inv(cov)
	cov_det = np.linalg.det(cov)
	a = np.delete(a,3,1)
	g = -0.5*((a-b).dot(cov_inv).dot((a-b).transpose()))[0][0] -0.5*np.log(cov_det) + np.log(p)
	return g
		
m1 = [0,0,0]
m2 = [1,2,2]
m3 = [3,3,4]

cov1 = [[0.8,0.2,0.1],[0.2,0.8,0.2],[0.1,0.2,0.8]]
cov2 = [[0.6,0.01,0.01],[0.01,0.8,0.01],[0.01,0.01,0.6]]
cov3 = [[0.6,0.1,0.1],[0.1,0.6,0.1],[0.1,0.1,0.6]]

#create the training dataset
xc1 = np.random.multivariate_normal(m1,cov1,1000).T
xc2 = np.random.multivariate_normal(m2,cov2,1000).T
xc3 = np.random.multivariate_normal(m3,cov3,1000).T
c1 = np.ones((1000,1))
c2 = 2*np.ones((1000,1))
c3 = 3*np.ones((1000,1))

xc1 = xc1.transpose()
xc2 = xc2.transpose()
xc3 = xc3.transpose()
C = np.concatenate((c1,c2,c3))
X = np.concatenate((xc1,xc2,xc3))
X = np.hstack((X,C))
np.random.shuffle(X)

#create test dataset
xc1t = np.random.multivariate_normal(m1,cov1,1000).T
xc2t = np.random.multivariate_normal(m2,cov2,1000).T
xc3t = np.random.multivariate_normal(m3,cov3,1000).T
c1t = np.ones((1000,1))
c2t = 2*np.ones((1000,1))
c3t = 3*np.ones((1000,1))

xc1t = xc1t.transpose()
xc2t = xc2t.transpose()
xc3t = xc3t.transpose()
Ct = np.concatenate((c1t,c2t,c3t))
Xt = np.concatenate((xc1t,xc2t,xc3t))
Xt = np.hstack((Xt,Ct))
np.random.shuffle(Xt)

#estimate mean
m1_estd = np.array([[0,0,0]])
m2_estd = np.array([[0,0,0]])
m3_estd = np.array([[0,0,0]])

for x in xc1:
	m1_estd[0][0] = m1_estd[0][0] + x[0]
	m1_estd[0][1] = m1_estd[0][1] + x[1]
	m1_estd[0][2] = m1_estd[0][2] + x[2]
for x in xc2:
	m2_estd[0][0] = m2_estd[0][0] + x[0]
	m2_estd[0][1] = m2_estd[0][1] + x[1]
	m2_estd[0][2] = m2_estd[0][2] + x[2]
for x in xc3:
	m3_estd[0][0] = m3_estd[0][0] + x[0]
	m3_estd[0][1] = m3_estd[0][1] + x[1]
	m3_estd[0][2] = m3_estd[0][2] + x[2]

m1_estd = m1_estd/1000
m2_estd = m2_estd/1000
m3_estd = m3_estd/1000

#estimate covariance matrices
cov1_estd = np.zeros((3,3))
cov2_estd = np.zeros((3,3))
cov3_estd = np.zeros((3,3))

for x in xc1:
	cov1_estd = cov1_estd + (np.array([x])-m1_estd).transpose().dot(np.array([x])-m1_estd)
for x in xc2:
	cov2_estd = cov2_estd + (np.array([x])-m2_estd).transpose().dot(np.array([x])-m2_estd)
for x in xc3:
	cov3_estd = cov3_estd + (np.array([x])-m3_estd).transpose().dot(np.array([x])-m3_estd)
	
cov1_estd = cov1_estd/999
cov2_estd = cov2_estd/999
cov3_estd = cov3_estd/999

#classify test points using bayesian classifier
predictions = []
for i in range(0,3000):
	g1 = classify(np.array([Xt[i]]),m1_estd,cov1_estd)
	g2 = classify(np.array([Xt[i]]),m2_estd,cov2_estd)
	g3 = classify(np.array([Xt[i]]),m3_estd,cov3_estd)
	
	if g1>g2 and g1>g3:
		predictions.append(1)
	elif g2>g3:
		predictions.append(2)
	else:
		predictions.append(3)

#calculate accuracy
correct = 0
for i in range(0,3000):
	if predictions[i] == Xt[i][3]:
		correct = correct+1
print('accuracy: ',(correct/3000)*100)		

#change prior probabilities
predictions2 = []
for i in range(0,3000):
	g1 = classify2(np.array([Xt[i]]),m1_estd,cov1_estd,0.125)
	g2 = classify2(np.array([Xt[i]]),m2_estd,cov2_estd,0.125)
	g3 = classify2(np.array([Xt[i]]),m3_estd,cov3_estd,0.750)
	
	if g1>g2 and g1>g3:
		predictions2.append(1)
	elif g2>g3:
		predictions2.append(2)
	else:
		predictions2.append(3)

#calculate accuracy
correct = 0
for i in range(0,3000):
	if predictions2[i] == Xt[i][3]:
		correct = correct+1
print('accuracy after changing prior probabilities: ',(correct/3000)*100)		



