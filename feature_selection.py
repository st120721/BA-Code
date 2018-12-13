import numpy as  np
import os
import  pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, NMF
import feature_extraction


class Feature_Selection:
    list_selection_methode =["Sequential Feature Selector"]

    def __init__(self,project_name,features,labels,fs_methode_name,ml_algorithm_name,score):

        self.project_name =project_name
        self.features =features
        self.labels =labels
        self.fs_methode_name =fs_methode_name
        self.ml_algorithm_name =ml_algorithm_name
        self.score =score
        self.output_path ="Result_Daten\\"+project_name



class Sequential_Feature_Selector():
    type_sequential_forward_selector = ["Sequential Forward Selection (SFS)", "Sequential Backward Selection (SBS)",
                                        "Sequential Forward Floating Selection(SFFS)",
                                        "Sequential Backward Floating Selection(SBFS)"]
    list_parameter = ["estimator", "k_features", "forward", "floating", "scoring", "cv"]
    param_grid = {'feature_selection__k_features': np.arange(1, 3, 1, dtype=int).tolist(),
                 #  'feature_selection__forward':[True,False],
                 # "feature_selection__floating":[True,False],
                  }

    @staticmethod
    def sfs_selectors( features, labels,classifier,score):

        # pre-processing
        # features = preprocessing.scale(features)
        # best_score = 0
        # for k in np.arange(1, 2, 1, dtype=int):
        #     classifier = KNeighborsClassifier(n_neighbors=k)
            feature_selector = SequentialFeatureSelector(classifier)
            pipe = Pipeline([('feature_selection', feature_selector), ('classifier', classifier)])
            param_grid=Sequential_Feature_Selector.param_grid
            # n_jobs = -1,
            grid = GridSearchCV(pipe, cv=5, param_grid=param_grid, scoring=score, iid=False, refit=True)
            grid.fit(features, labels)
            # if grid.best_score_ > best_score:
            #     best_score = grid.best_score_
            #     best_parameters = grid.best_params_
            #     best_estimator = grid.best_estimator_
            #     steps_for_k_feature = grid.best_estimator_.steps
            #
            #     print(best_score)
            #     print(best_parameters)
            #     print(best_estimator)
            #     print(steps_for_k_feature[0][1].k_feature_idx_)
            #     exit()

            return grid



# path ="Result_Daten\\test\\"+"FE_After_DWT.csv"
# data =pd.read_csv(path)
# features = data.drop(["label"],axis=1)
# labels = pd.DataFrame(data.ix[:, data.shape[1] - 1]).values.ravel()
# test =Sequential_Feature_Selector()
# test.sfs_selectors(features=features,labels=labels,classifier="knn",score="accuracy")
