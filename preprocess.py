"""This file produces the X and Y matrices for training and testing arrays and preprocesses them """

#required imports
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

#loading datafile
data = pd.read_csv('finalData.csv').values

#create dictionary for genres
genre_dict ={
    'Electronic' : 0,
    'Pop' : 1,
    'Experimental' :2,
    'Industrial' :3,
    'World': 4,
    'Latin':5,
    'Hip-hop': 6,
    'Rap' : 7,
    'Electrohouse' : 8,
    'Folk' : 9,
    'Rock' : 10,
    'Reggae' : 11,
    'Lo-fi' :12,
    'Instrumental' : 13,
    'Soundtrack' : 14,
}

#shuffle data
#data = shuffle(data)

#splicing unnnecessary columns
data = data[:,1:31]

#getting columns and features
genre_list = data[:,-1]
features = data[:,:-1]

#creating y vector
filtered_indices = [i for i, genre in enumerate(genre_list) if genre_dict.get(genre) is not None]
genre_list = [genre_dict.get(genre_list[i]) for i in filtered_indices]
features = [features[i] for i in filtered_indices]
#print(genre_list)
#dividing into training and test arrays
X_train, X_test, y_train, y_test = train_test_split(features, genre_list, test_size= 0.1)
X_train, X_val, y_train, y_val = train_test_split(features, genre_list, test_size = 0.1)

# test to see which num_components value I need
'''
pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance Ratio by Number of Components')
plt.grid(True)
plt.show()

# Do PCA
num_components = 10
pca = PCA(n_components=num_components)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
X_val_pca = pca.transform(X_val)
'''
#standardizing features
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
X_val_scaled = preprocessing.scale(X_val)

#function to return values when called
def return_data():
    return X_train_scaled,X_test_scaled,X_val_scaled,y_train,y_test,y_val
