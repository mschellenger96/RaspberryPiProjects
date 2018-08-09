from sklearn.naive_bayes import GaussianNB
import numpy as np 
import matplotlib.pyplot as plt 

a = []
b = []
in_file = 'data_multivar.txt'

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

def plot_classification(classification, a, b):
	a_min, a_max = min(a[:,0]) - 1.0, max(a[:,0]) + 1.0
	b_min, b_max = min(a[:,1]) - 1.0, max(a[:,1]) + 1.0

	step_size = 0.01

	a_values, b_values = np.meshgrid(np.arange(a_min, a_max, step_size), np.arange(b_min, b_max, step_size)) 
	mesh_output_1 = classification.predict(np.c_[a_values.ravel(), b_values.ravel()])
	mesh_output_2 = mesh_output_1.reshape(a_values.shape)

	plt.figure()
	plt.pcolormesh(a_values, b_values, mesh_output_2, cmap=plt.cm.gray)
	plt.scatter(a[:,0], a[:,1], c=b, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

	plt.xlim(a_values.min(), a_values.max())
	plt.ylim(b_values.min(), b_values.max())

	plt.xticks((np.arange(int (min(a[:,0])-1), int(max(a[:,0])+1), 1)))
	plt.xticks((np.arange(int (min(a[:,1])-1), int(max(a[:,1])+1), 1)))
	plt.show()

plot_classification(classification, a, b)




