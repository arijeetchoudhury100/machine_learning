import numpy as np
import matplotlib.pyplot as plt

def fact(x):
	f = 1
	x = int(x)
	for i in range(1,x+1):
		f = f*i
	return f;
def B(a,b):
	res = (fact(a-1)*fact(b-1))/fact(a+b-1)
	return res
def gamma(X,a,b):
	y = []
	for x in X:
		y.append((x**(a-1) * (1-x)**(b-1))/B(a,b))
	return y

X = np.arange(0,6,0.05)
y = gamma(X,0,1)

plt.plot(X,y)
plt.show()
