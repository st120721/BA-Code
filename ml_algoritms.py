import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, NMF
import feature_extraction
import feature_selection

class ML_Algoritms:
    list_ml_algorithms =["Support Vector Machines(SVM)","K Nearest Neighbors(KNN)","Random Forests(RF)"]
    score =["accuracy","average precison"]

    # def standardization(self,data):
    #     return preprocessing.scale(data)


class SVM():
    name = "Support Vector Machines(SVM)"
    c_parameter = np.arange(0.1, 10, 0.5, dtype=float)
    gamma_parameter = np.arange(0.01, 1, 0.05, dtype=float)
    parameters_grid = {'C': c_parameter,'gamma': gamma_parameter}

class KNN():
    name ="K Nearest Neighbors(KNN)"
    k_parameter = np.arange(1, 2, 1, dtype=int)
    parameters_grid = {'classifier__n_neighbors': k_parameter}

    def tunning_parameter_sfs(self,features,labels,score):

        # pre-processing
        features = preprocessing.scale(features)

        best_score = 0
        for k in self.k_parameter:

            clf=KNeighborsClassifier(n_neighbors=k)

            grid =feature_selection.Sequential_Feature_Selector().sfs_selectors(features, labels,clf,score)
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_parameters = grid.best_params_
                best_estimator = grid.best_estimator_
                steps_for_k_feature = grid.best_estimator_.steps

                print(best_score)
                print(best_parameters)
                print(best_estimator)
                print(steps_for_k_feature[0][1].k_feature_idx_)
                exit()
        return


class RF():
    n_estimators_parameter = np.arange(50, 100, 1, dtype=int)
    parameters_grid= {'n_estimators': n_estimators_parameter}

class Scroce():
    score = ["accuracy", "average precison"]




path ="Result_Daten\\test\\"+"FE_After_DWT.csv"
data =pd.read_csv(path)
features = data.drop(["label"],axis=1)
labels = pd.DataFrame(data.ix[:, data.shape[1] - 1]).values.ravel()
ml_test =KNN().tunning_parameter_sfs(features=features,labels=labels,score="accuracy")


