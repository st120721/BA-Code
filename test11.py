import pandas as pd
from hyperopt import hp
import sklearn.datasets
from keras.utils import np_utils
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import hyperopt

#
iris = datasets.load_iris()
features=pd.DataFrame(iris.data).values
labels = iris.target
# print(features)
# print(labels)

# train test split 使用保证训练集和测试集比例相同
skf = StratifiedShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7)
print(skf.split(features,labels))
for train_index, test_index in skf.split(features, labels):

    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
y_train=y_train+1
print(y_train)
y_train =np_utils.to_categorical(y_train, num_classes=3)

print(y_train)