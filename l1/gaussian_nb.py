import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) # features
Y = np.array([1, 1, 1, 2, 2, 2]) # labels

# training
classifier = GaussianNB()
classifier.fit(X, Y)

# prediction
print(classifier.predict([[-0.8, -1]]))
