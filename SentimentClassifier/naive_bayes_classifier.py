from sklearn.naive_bayes import GaussianNB
import numpy as np 
import matplotlib.pyplot as plt 

import warnings


in_file = 'data_multivar.txt'

a = []
b = []

warnings.filterwarnings("ignore")
with open(in_file, 'r') as file:
	for line in file.readlines():
		data = [float(x) for x in line.split(',')]
		a.append(data[:-1])
		b.append(data[-1])
a = np.array(a)
b = np.array(b)

classification = GaussianNB()
classification.fit(a, b)
b_pred = classification.predict(a)

correctness = 100.0 * (b == b_pred).sum() / a.shape[0]
print ("correctness of the classification =", round(correctness, 2), "%")
