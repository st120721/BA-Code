import pandas as pd
import numpy as np
import os
import datetime as d
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

    def wirte_to_csv_txt(self,output_data):
        path = self.output_path
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        output_name ="FS_After_"+self.fs_methode_name+\
                     "for"+self.ml_algorithm_name+"("+self.score+")"+".csv"
        path=path+"\\"+output_name
        output_data.to_csv(path,index=False)

class SVM():
    name = "Support Vector Machines(SVM)"
    c_parameter = np.arange(0.1, 10, 0.5, dtype=float)
    gamma_parameter = np.arange(0.01, 1, 0.05, dtype=float)
    parameters_grid = {'C': c_parameter,'gamma': gamma_parameter}

    def tunning_parameter_sfs(self, features, labels, score):

        # pre-processing
        features = preprocessing.scale(features)

        best_score = 0
        for C in self.c_parameter:
            for gamma in self.gamma_parameter:
                clf = SVC(C=C,gamma=gamma)
                grid = feature_selection.Sequential_Feature_Selector().sfs_selectors(features, labels, clf, score)
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_parameters = grid.best_params_
                    best_estimator = grid.best_estimator_
                    selected_k_feature = grid.best_estimator_.steps

        return best_score, best_parameters, best_estimator, selected_k_feature

class KNN():
    name ="K Nearest Neighbors(KNN)"
    k_parameter = np.arange(1, 2, 1, dtype=int)
    parameters_grid = {'classifier__n_neighbors': k_parameter}

    def __init__(self,project_name,fs_methode,features,labels,score):
        self.project_name =project_name
        self.time_start =d.datetime.now()

        self.fs_methode=fs_methode
        self.features=features
        self.labels=labels
        self.score =score
        self.best_score, self.best_parameters,self.best_n_neighbors,self.best_estimator, self.selected_features=self.tunning_parameter_sfs()
        self.output_path ="Result_Daten\\"+project_name
        self.time_end =d.datetime.now()
        self.dauer=(self.time_end-self.time_start)
        self.wirte_info_to_txt()
        self.wirte_selected_feature_to_csv()

    def wirte_info_to_txt(self):

        info = [self.project_name,
                "\nstart time: "+str(self.time_start.strftime("%d.%m.%Y-%H:%M:%S")),
                "\nend time: "+str(self.time_end.strftime("%d.%m.%Y-%H:%M:%S")),
                "\nduration: "+str(self.dauer),
                "\nscore: "+str(self.score),
                "\nscore: "+str(self.best_score),
                "\nparameter of SFS: " + str(self.best_parameters),
                "\nn neighbors: "+str(self.best_n_neighbors),
                "\nindex of selected features: " + str(self.selected_features),
                "\nestimator: "+str(self.best_estimator),
                "\n\n"
                ]
        path = self.output_path
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        output_name ="KNN_"+self.fs_methode+"_"+self.score+".txt"
        path=path+"\\"+output_name

        with open(path, 'x' and "a") as f:
            for i in info:
                f.write(i)

    def wirte_selected_feature_to_csv(self):
        path = self.output_path
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        output_name = "FS_with_" + self.fs_methode + ".csv"
        path = path + "\\" + output_name
        features=self.features.iloc[:,list(self.selected_features)]
        features.to_csv(path, index=False)

    def tunning_parameter_sfs(self):
        features=self.features
        labels=self.labels
        score=self.score
        # pre-processing
        features = preprocessing.scale(features)

        best_score = 0
        for k in self.k_parameter:

            clf=KNeighborsClassifier(n_neighbors=k)
            grid =feature_selection.Sequential_Feature_Selector.sfs_selectors(features, labels,clf,score)
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_parameters = grid.best_params_
                best_n_neighbors=k
                best_estimator = grid.best_estimator_
                selected_k_feature = grid.best_estimator_.steps[0][1].k_feature_idx_
                # print(best_score)
                # print(best_parameters)
                # print(best_estimator)
                # print(selected_k_feature[0][1].k_feature_idx_)

        return  best_score,best_parameters,best_n_neighbors,best_estimator,selected_k_feature



class RF():
    n_estimators_parameter = np.arange(50, 100, 1, dtype=int)
    parameters_grid= {'n_estimators': n_estimators_parameter}

class Scroce():
    score = ["accuracy", "average precison"]




path ="Result_Daten\\test\\"+"FE_After_DWT.csv"
data =pd.read_csv(path)
features = data.drop(["label"],axis=1)
labels = pd.DataFrame(data.ix[:, data.shape[1] - 1]).values.ravel()
ml_test =KNN(project_name="test",fs_methode="sfs",features=features,labels=labels,score="accuracy")


