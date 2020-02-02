import numpy as np
import matplotlib.pyplot as plt

def fact(x):
	f = 1
	x = int(x)
	for i in range(1,x+1):
		f = f*i
	return f;
def gamma(X,a,b):
	y = []
	for x in X:
		y.append((b**a * x**(a-1) * np.e**(-1*b*x))/fact(a-1))
	return y

X = np.arange(2,100)
y = gamma(X,0.1,0.1)

plt.plot(X,y)
plt.show()
