import numpy as np

def bayesian_classifier(x,cov,mean):
	inv_cov = np.linalg.inv(cov)
	det_cov = np.linalg.det(cov)
	g = -0.5*((x-mean).dot(inv_cov).dot((x-mean).transpose())) - 0.5*np.log(det_cov)
	return g

iris_data = open('iris.data','r')
iris_data_list = []

for line in iris_data:
	iris_data_list.append(line.split(','))

#prepare training and testing data
iris_data_list.pop()
iris_data_array = np.array(iris_data_list)
np.random.shuffle(iris_data_array)

iris_data_X = iris_data_array[:,0:4]
iris_data_y = iris_data_array[:,4]
iris_data_X = iris_data_X.astype(np.float)

#training data
iris_data_X_train = iris_data_X[0:100,:]
iris_data_y_train = iris_data_y[0:100]

#testing data
iris_data_X_test = iris_data_X[100:150,:]
iris_data_y_test = iris_data_y[100:]
m1 = np.zeros((1,4))	#iris-versicolor
m2 = np.zeros((1,4))	#iris-virginica
m3 = np.zeros((1,4))	#iris-setosa

c1 = 0
c2 = 0
c3 = 0

prior1 = 0
prior2 = 0
prior3 = 0
#calculate mean
for i in range(0,len(iris_data_X_train)):
	if iris_data_y_train[i] == 'Iris-versicolor\n':
		m1[0][0] = m1[0][0] + iris_data_X_train[i][0]
		m1[0][1] = m1[0][1] + iris_data_X_train[i][1]
		m1[0][2] = m1[0][2] + iris_data_X_train[i][2]
		m1[0][3] = m1[0][3] + iris_data_X_train[i][3]
		c1 = c1+1
	elif iris_data_y_train[i] == 'Iris-virginica\n':
		m2[0][0] = m2[0][0] + iris_data_X_train[i][0]
		m2[0][1] = m2[0][1] + iris_data_X_train[i][1]
		m2[0][2] = m2[0][2] + iris_data_X_train[i][2]
		m2[0][3] = m2[0][3] + iris_data_X_train[i][3]
		c2 = c2+1
	elif iris_data_y_train[i] == 'Iris-setosa\n':
		m3[0][0] = m3[0][0] + iris_data_X_train[i][0]
		m3[0][1] = m3[0][1] + iris_data_X_train[i][1]
		m3[0][2] = m3[0][2] + iris_data_X_train[i][2]
		m3[0][3] = m3[0][3] + iris_data_X_train[i][3]
		c3 = c3+1
m1 = m1/c1
m2 = m2/c2
m3 = m3/c3

prior1 = c1/(c1+c2+c3)
prior2 = c2/(c1+c2+c3)
prior3 = c3/(c1+c2+c3)
#calculate covariance matrices
cov1 = np.zeros((4,4))
cov2 = np.zeros((4,4))
cov3 = np.zeros((4,4))

for i in range(0,len(iris_data_X_train)):
	if iris_data_y_train[i] == 'Iris-versicolor\n':
		cov1 = cov1 + (np.array([iris_data_X_train[i]])-m1).transpose().dot(np.array([iris_data_X_train[i]])-m1)
	elif iris_data_y_train[i] == 'Iris-virginica\n':
		cov2 = cov2 + (np.array([iris_data_X_train[i]])-m2).transpose().dot(np.array([iris_data_X_train[i]])-m2)
	elif iris_data_y_train[i] == 'Iris-setosa\n':
		cov3 = cov3 + (np.array([iris_data_X_train[i]])-m3).transpose().dot(np.array([iris_data_X_train[i]])-m3)
cov1 = cov1/c1
cov2 = cov2/c2
cov3 = cov3/c3
#classify points
predictions = []
for i in range(0,len(iris_data_X_test)):
	g1 = bayesian_classifier(np.array([iris_data_X_test[i]]),cov1,m1)
	g2 = bayesian_classifier(np.array([iris_data_X_test[i]]),cov2,m2)
	g3 = bayesian_classifier(np.array([iris_data_X_test[i]]),cov3,m3)

	if g1 > g2 and g1 > g3:
		predictions.append('Iris-versicolor\n')
	elif g2 > g3:
		predictions.append('Iris-virginica\n')
	else:
		predictions.append('Iris-setosa\n')

correct = 0
for i in range(0,len(predictions)):
	if predictions[i] == iris_data_y_test[i]:
		correct = correct + 1
print('Accuracy:',(correct/len(predictions))*100)