import pandas as pd
import numpy as np
import os
import datetime as d
import csv
from feature_selection_hyperopt import Sequential_Feature_Selector
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import preprocessing,datasets
from sklearn.metrics import make_scorer, fbeta_score, recall_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import feature_selection
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import PCA, NMF
import feature_extraction
import feature_selection


class ML_Algoritms:
    list_ml_algorithms = ["Support Vector Machines(SVM)", "K Nearest Neighbors(KNN)",
                          "Random Forests(RF)", "Bayes()", "Artificial Neural Network(ANN)"]
    score = ["accuracy"]

    def __init__(self, test_idx, classifier,features, labels, scoring):
        if classifier == "KNN":
            clf = KNN(test_idx=test_idx,features=features, labels=labels, scoring=scoring)

        self.wirte_result_to_txt(clf)
        self.wirte_selected_feature_to_csv(clf)

    @staticmethod
    def wirte_result_to_txt(clf):
        folder_name = "Result_Daten\\" + clf.test_idx
        isExists = os.path.exists(folder_name)
        if not isExists:
            os.makedirs(folder_name)

        file_name =clf.name+"_"+str(clf.scoring)+".txt"
        output_path=folder_name+"\\"+file_name
        with open(output_path, 'x' and "a") as f:
            for key, value in clf.result.items():
                info = key+": "+str(value)
                f.write(info)

    @staticmethod
    def wirte_selected_feature_to_csv(clf):

        folder_name = "Result_Daten\\" + clf.test_idx
        isExists = os.path.exists(folder_name)
        if not isExists:
            os.makedirs(folder_name)

        file_name = "selected_features_"  + str(clf.scoring) + ".csv"
        output_path = folder_name+"\\" + file_name
        features = clf.features.iloc[:, list(clf.result[ "selected_features_idx"])]
        features.to_csv(output_path, index=False)

class KNN():
    name = "KNN"
    k_parameter = np.arange(1, 10, 1, dtype=int)
    parameters_grid = {'n_neighbors': hp.choice('n_neighbors', k_parameter.tolist())}

    def __init__(self, test_idx, features, labels, scoring,loops=100):
        self.test_idx = test_idx
        self.features = features
        self.labels = labels
        self.scoring = scoring
        self.loops = loops
        self.result= self.tuning_parameter()
      

    def tuning_parameter(self):
        features =self.features
        labels=self.labels

        iris = datasets.load_iris()
        features = iris.data
        labels = iris.target

        # File to save results
        folder_name ="Result_Daten\\" + self.test_idx
        isExists = os.path.exists(folder_name)
        if not isExists:
            os.makedirs(folder_name)

        file_name = self.name + "_" + str(self.scoring) + "_" + "tunning_parameter_info.csv"

        writer_path = folder_name+ "\\" + file_name
        with open(writer_path, "x" and "a")as f:
            writer = csv.writer(f)
            writer.writerow(["loop", "SFS","num selected features", "feature idx", str(self.scoring),"n_neighbors"])


        # pre-processing
        features = preprocessing.scale(features)

        # parameter space
        sfs_param_space = Sequential_Feature_Selector.sfs_param_space
        knn_param_space = self.parameters_grid
        parma_space = dict(sfs_param_space, **knn_param_space)

        # loop
        temp_loop=0
        result={
        "time start":  d.datetime.now(),
        "time end":None,
        "time dauer": None,
        "loops":0,
        "scoring":self.scoring,
        "best_score" : 0,
        "sfs_type" : "",
        "num_features" : 0,
        "selected_features_idx" : [],
        "best_n_neighbors" : 0
        }

        def tuning_parameter_one_loop(params):



            temp_loop = 1
            print("start tunning param")
            print(temp_loop)




            #knn
            n_neighbors = params['n_neighbors']
            clf = KNeighborsClassifier(n_neighbors)

            #sfs
            k_features = params["k_features"]
            foward = params["forward"]
            floating = params["floating"]


            sfs_type =Sequential_Feature_Selector.sfs_definition(forward=foward,floating=floating)

            print(k_features)
            print(sfs_type)
            print(n_neighbors)

            sfs = SequentialFeatureSelector(estimator=clf, k_features=k_features,
                                            forward=foward, floating=floating, scoring=self.scoring,
                                            n_jobs=-1,
                                            cv=5, )
            sfs.fit(features, labels)
            print(k_features)
            print(sfs_type)
            print(n_neighbors)
            print("end one loop")

            #  Write info to the file
            with open(writer_path, "a")as f:
                writer = csv.writer(f)
                writer.writerow([temp_loop, sfs_type, sfs.k_features, sfs.k_feature_idx_, sfs.k_score_,n_neighbors])


            return sfs.k_score_,sfs_type,k_features,sfs.k_feature_idx_,n_neighbors

        def scoring(params):
            score, sfs_type, k_features,feature_idx_, n_neighbors =tuning_parameter_one_loop(params)
            # global  best_score,sfs_type,num_features,selected_features_idx,best_n_neighbors
            if score>result["best_score"]:

                result["best_score"]=score
                result["sfs_type"]=sfs_type
                result["num_features"]=k_features
                result["selected_features_idx"]=feature_idx_
                result["best_n_neighbors"]=n_neighbors

            return {'loss': -score, 'status': STATUS_OK}



        trials = Trials()
        fmin(fn=scoring, space=parma_space, algo=tpe.suggest,
                    max_evals=self.loops,
                    trials=trials)

       
        result["time end"]=d.datetime.now()
        result["time end"]=result["time end"]-result["time start"]
        result["loops"]=temp_loop
        return result


class Scoring():
    score = ["accuracy", "average precison"]
    f2_score = make_scorer(fbeta_score, beta=2)
    recall_16 = make_scorer(recall_score, labels=[16], average=None)
    recall_macro = make_scorer(recall_score, average="macro")


path ="Result_Daten\\KNN_accuracy_14.12.2018_13.00.00\\"+"FE_WP(TestData_1700.csv).csv"
data =pd.read_csv(path)
features = data.drop(["label"],axis=1)
labels = pd.DataFrame(data.ix[:, data.shape[1] - 1]).values.ravel()
ml_test =KNN(test_idx="test",features=features,labels=labels,scoring="accuracy")


