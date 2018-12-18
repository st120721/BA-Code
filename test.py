import pandas as pd

from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from feature_extraction import Feature_Extraction
from ml_algoritms import  NN

path ="Result_Daten\\KNN_accuracy_14.12.2018_13.07.11\\"+"FS_features_SFS_accuracy.csv"
path ="Result_Daten\\KNN_accuracy_14.12.2018_13.07.11\\"+"FE_WP(TestData_1700.csv).csv"
features =pd.read_csv(path)
features = features.drop(["label"],axis=1)
features = preprocessing.scale(features)
print (features.shape)
path ="Result_Daten\\KNN_accuracy_14.12.2018_13.07.11\\"+"FE_WP(TestData_1700.csv).csv"
labels =pd.read_csv(path)
labels = pd.DataFrame(labels.ix[:, labels.shape[1] - 1])
labels =labels-1
print(labels.shape)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
y_train = np_utils.to_categorical(y_train, num_classes=17)
y_test = np_utils.to_categorical(y_test, num_classes=17)
NN().model(features=X_train,labels=y_train)