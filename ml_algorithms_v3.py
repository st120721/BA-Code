import pandas as pd
import numpy as np
import os
import datetime as d
import csv
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import preprocessing, datasets
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


class ML_Algorithms:
    type_sequential_feature_selector = ["Sequential Forward Selection (SFS)", "Sequential Backward Selection (SBS)",
                                        "Sequential Forward Floating Selection(SFFS)",
                                        "Sequential Backward Floating Selection(SBFS)"]
    sfs_param_space = {"k_features": hp.choice("k_features", np.arange(1, 3, 1, dtype=int).tolist()),
                       "forward": hp.choice("forward", [True, False]),
                       "floating": hp.choice("floating", [True, False]),
                       }

    list_ml_algorithms = ["Support Vector Machines(SVM)", "K Nearest Neighbors(KNN)",
                          "Random Forests(RF)", "Bayes()", "Artificial Neural Network(ANN)"]
    score = ["accuracy"]

    knn_dict = dict(
        name="KNN",
        parameters_list=["n_neighbors"],
        parameters_grid={'n_neighbors': hp.choice('n_neighbors', np.arange(1, 10, 1, dtype=int).tolist())},
        estimator=KNeighborsClassifier(),
    )
    svm_dict=dict(
    name="SVM",
    parameters_list=["kernel","C","gamma"],
    parameters_grid={'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
                     'C': hp.uniform('C', 0, 20),
                    'gamma': hp.uniform('gamma', 0, 20),},
    estimator = SVC(),

    )
    rf_dict=dict(
        name ="RF",
        parameters_list=['max_depth', 'max_features', 'n_estimators', 'criterion'],
        parameters_grid={ 'max_depth': hp.choice('max_depth', range(1, 20)),
                          'max_features': hp.choice('max_features', range(1, 5)),
                          'n_estimators': hp.choice('n_estimators', range(1, 20)),
                           'criterion': hp.choice('criterion', ["gini", "entropy"])},
        estimator = RandomForestClassifier()
    )
    
    algorithms_dict =dict(
        KNN=knn_dict,
        SVM=svm_dict,
        RF=rf_dict,
        
    )
    def __init__(self,algorithm,loops, features, labels, scoring,folder_path=""):
        self.algorithm=algorithm
        self.features = features
        self.labels = labels
        self.scoring = scoring
        self.loops = loops


        self.folder_path ="Result_Daten\\" +  folder_path

        self.result = self.tuning_parameter()
        self.wirte_result_to_txt()

        self.wirte_selected_feature_to_csv()

    
    def wirte_result_to_txt(self):
        folder_path = self.folder_path
        isExists = os.path.exists(folder_path)
        if not isExists:
            os.makedirs(folder_path)

        file_name = self.algorithm + "_" + str(self.scoring) + ".txt"
        output_path = folder_path + "\\" + file_name
        with open(output_path, 'w'and"a" ) as f:
            for key, value in self.result.items():
                info = key + ": " + str(value)+"\n"
                f.write(info)

    
    def wirte_selected_feature_to_csv(self):


        folder_path = self.folder_path
        isExists = os.path.exists(folder_path)
        if not isExists:
            os.makedirs(folder_path)

        file_name = "selected_features_" + str(self.scoring) + ".csv"
        output_path = folder_path + "\\" + file_name
        features = self.features.iloc[:, list(self.result["selected_features_idx"])]
        features.to_csv(output_path, index=False)

    def tuning_parameter(self):

        algorithm_dict = self.algorithms_dict[self.algorithm]
        scoring = self.scoring
        folder_path = self.folder_path

        # File to save results
        # folder_path = self.folder_path
        isExists = os.path.exists(folder_path)
        if not isExists:
            os.makedirs(folder_path)
        file_name = self.algorithm + "_" + str(scoring) + "_" + "tunning_parameter_info.csv"
        writer_path = folder_path + "\\" + file_name
        with open(writer_path, "x" and "w")as f:
            writer = csv.writer(f)
            loop_info_list = ["loop",  "SFS", "num selected features",
                              "feature idx"]+(algorithm_dict["parameters_list"])+[str(scoring)],

            writer.writerow(loop_info_list)

        # pre-processing
        features = preprocessing.scale(self.features)
        labels =self.labels

        # parameter space
        sfs_param_space = self.sfs_param_space
        algorithm_param_space = algorithm_dict["parameters_grid"]
        parma_space = dict(sfs_param_space, **algorithm_param_space)

        # loop
        temp_loop = 0

        result = {
            "time start": d.datetime.now(),
            "time end": None,
            "duration": None,
            "loops": 0,
            "self.scoring": scoring,
            "best_score": 0,
            "sfs_type": None,
            "num_features": 0,
            "selected_features_idx": None,
        }
        result = dict(result, **(dict.fromkeys(algorithm_dict["parameters_list"], None)))

        print(loop_info_list)
        def tuning_parameter_one_loop(params):

            nonlocal temp_loop
            temp_loop = temp_loop + 1
            print("loop: ",temp_loop)
            # knn
            # n_neighbors = params['n_neighbors']
            clf = algorithm_dict["estimator"]


            for keys1, values1 in params.items():

                for keys2, values2 in clf.__dict__.items():
                    if keys1 == keys2:
                        clf.__dict__[keys1] = values1


            # sfs
            sfs = SequentialFeatureSelector(estimator=clf, scoring=scoring,
                                              n_jobs=-1,
                                              cv=5, )
            for keys1, values1 in params.items():
                for keys2, values2 in sfs.__dict__.items():
                    if keys1 == keys2:
                        sfs.__dict__[keys1] = values1

            forward=sfs.forward
            floating =sfs.floating
            if forward == True and floating == False:
                sfs_type = "SFS"
            elif forward == False and floating == False:
                sfs_type = "SBS"
            elif forward == True and floating == True:
                sfs_type = "FSFS"
            elif forward == False and floating == True:
                sfs_type = "FSBS"


            sfs.fit(features, labels)
            score = sfs.k_score_


            loop_info = [temp_loop, sfs_type, sfs.k_features, sfs.k_feature_idx_]
            for pa in algorithm_dict["parameters_list"]:
                    loop_info.append(clf.__getattribute__(pa))

            print(sfs_type)
            print(params)
            print(clf.__dict__)
            print(loop_info)
            print(score)

            #  Write info to the file
            with open(writer_path, "a")as f:
                writer = csv.writer(f)
                writer.writerow(loop_info)

            # global  best_score,sfs_type,num_features,selected_features_idx,best_n_neighbors
            if score > result["best_score"]:
                result["best_score"] = score
                result["sfs_type"] = sfs_type
                result["num_features"] = sfs.k_features
                result["selected_features_idx"] = sfs.k_feature_idx_
                for keys1, values1 in params.items():
                    for key2, values2 in result.items():
                        if keys1 == key2:
                            result[keys1] = values1

                print("best")
            print(result)
            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        fmin(fn=tuning_parameter_one_loop, space=parma_space, algo=tpe.suggest,
             max_evals=self.loops,
             trials=trials)


        result["time end"] = d.datetime.now()
        result["duration"] = result["time end"] - result["time start"]
        result["loops"] = temp_loop

        print(result)
        return result

    
       

    

class Scoring():
    score = ["accuracy", "average precison"]
    f2_score = make_scorer(fbeta_score, beta=2)
    recall_16 = make_scorer(recall_score, labels=[16], average=None)
    recall_macro = make_scorer(recall_score, average="macro")



iris = datasets.load_iris()
features=pd.DataFrame(iris.data)
labels = iris.target

test =ML_Algorithms("RF",10, features, labels, "accuracy",folder_path="test")

