# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Credit_Card_Applications.csv");

X = dataset.iloc[:, :-1].values # get all index other than the last column
y = dataset.iloc[:, -1].values # only last variable

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # defining the range of normalization
X = sc.fit_transform(X) #fit_tranform to normalize the variables

# Training the SOM
# either we write this from SOM
# or take it from other developers MiniSom 1.0
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

#initialize random weight
som.random_weights_init(X)

#train the dataset
som.train_random(data = X, num_iteration = 100)

#vizualize
#MID mean to neuron distance - higher the MID -> more the winning node is far away(outlier)
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T) #will return all te mean to neuro distances into 1 matrix
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X): # i is the indexes ,x will be all the custmers in index
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)
