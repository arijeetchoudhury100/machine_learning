import numpy as np
import matplotlib.pyplot as plt

def exponential(X,lamb):
	y = []
	for x in X:
		if x < 0:
			y.append(0)
		else:
			y.append(lamb*np.e**(-1*lamb*x))
	return y

X = np.arange(0,10,0.05)
y = exponential(X,0.5)
plt.plot(X,y)
plt.show()
