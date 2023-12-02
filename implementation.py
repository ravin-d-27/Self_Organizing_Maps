import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv("Credit_card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print("Shape: ", dataset.shape)

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)

# Implementing SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len=15, learning_rate=0.5, sigma=1.0) # Sigma is the radius
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# Visualizing the Results
# We need to get the MID (Mean Interneuron Distance)

from pylab import bone, pcolor, plot, show, colorbar
bone() # to initialize the window
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)

show()

# Catching the Frauds

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,1)], mappings[(4,1)]), axis = 0)
