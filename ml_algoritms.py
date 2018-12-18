import pandas as pd
import numpy as np
import os
import datetime as d
from sklearn import preprocessing
from sklearn.metrics import make_scorer, fbeta_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, NMF
import feature_extraction
import feature_selection

class ML_Algoritms:
    list_ml_algorithms =["Support Vector Machines(SVM)","K Nearest Neighbors(KNN)",
                         "Random Forests(RF)","Bayes()","Artificial Neural Network(ANN)"]
    score =["accuracy"]


    def __init__(self,test_idx,classifier, fs_methode, features, labels, score):
        if classifier=="KNN":
            clf=KNN(test_idx=test_idx, fs_methode=fs_methode, features=features, labels=labels, score=score)
        elif classifier=="SVM":
            clf=SVM(test_idx=test_idx, fs_methode=fs_methode, features=features, labels=labels, score=score)
        elif classifier=="RF":
            clf=RF(test_idx=test_idx, fs_methode=fs_methode, features=features, labels=labels, score=score)

        ML_Algoritms.wirte_info_to_txt(clf)
        ML_Algoritms.wirte_selected_feature_to_csv(clf)

    @staticmethod
    def wirte_info_to_txt(clf):

        info = [clf.test_idx,
                "\nstart time: "+str(clf.time_start.strftime("%d.%m.%Y-%H:%M:%S")),
                "\nend time: "+str(clf.time_end.strftime("%d.%m.%Y-%H:%M:%S")),
                "\nduration: "+str(clf.dauer),
                "\nscoring: "+str(clf.score),
                "\nbest score: "+str(clf.best_score),
                "\nparameter of SFS: " + str(clf.best_parameters_FS),
                "\nparameter of Classifier: "+str(clf.best_parameters_clf),
                "\nindex of selected features: " + str(clf.selected_features),
                "\nestimator: "+str(clf.best_estimator),
                "\n\n"
                ]
        path = clf.output_path
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        output_name =clf.name+"_"+clf.fs_methode+"_"+str(clf.score)+".txt"
        path=path+"\\"+output_name
        with open(path, 'x' and "a") as f:
            for i in info:
                f.write(i)

    @staticmethod
    def wirte_selected_feature_to_csv(clf):
        path = clf.output_path
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        output_name = "FS_features_" + clf.fs_methode + "_" +str(clf.score) + ".csv"
        path = path + "\\" + output_name
        features=clf.features.iloc[:,list(clf.selected_features)]
        features.to_csv(path, index=False)
        
class KNN():
    name ="KNN"
    k_parameter = np.arange(1, 2, 1, dtype=int)
    parameters_grid = {'classifier__n_neighbors': k_parameter}

    def __init__(self,test_idx,fs_methode,features,labels,score):
        self.test_idx =test_idx
        self.time_start =d.datetime.now()
        self.fs_methode=fs_methode
        self.features=features
        self.labels=labels
        self.score =score
        self.output_path = "Result_Daten\\" + test_idx
        self.best_score, self.best_parameters_FS,self.best_parameters_clf,self.best_estimator, self.selected_features=self.tunning_parameter_sfs()
        self.time_end =d.datetime.now()
        self.dauer=(self.time_end-self.time_start)
        # self.wirte_info_to_txt()
        # self.wirte_selected_feature_to_csv()

    def tunning_parameter_sfs(self):
        features=self.features
        labels=self.labels

        # pre-processing
        features = preprocessing.scale(features)

        best_score = 0
        run_info =pd.DataFrame()
        loop=0
        for k in self.k_parameter:
            loop =loop+1
            print("loop:",loop)
            clf=KNeighborsClassifier(n_neighbors=k)

            k_info = pd.DataFrame([k], columns=["k"])
            print("KNN k= ", k, "start")
            grid =feature_selection.Sequential_Feature_Selector.sfs_selectors(features, labels,clf,self.score)
            run_info_temp= pd.concat((k_info, pd.DataFrame(grid.cv_results_)), axis=1)
            run_info=pd.concat((run_info,run_info_temp),axis=0)

            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_parameters_FS= grid.best_params_
                best_parameters_clf=[k]
                best_estimator = grid.best_estimator_
                selected_k_feature = grid.best_estimator_.steps[0][1].k_feature_idx_
            print("KNN k= ", k, "end")
        output_name=self.output_path+"\\"+self.name+"_"+ self.fs_methode + "_"+str(self.score)+"_tunning_parameter_info.csv"
        run_info.to_csv(output_name)
        return  best_score,best_parameters_FS,best_parameters_clf,best_estimator,selected_k_feature

class SVM():
    name = "SVM"
    c_parameter = np.arange(0.1, 10, 0.5, dtype=float)
    gamma_parameter = np.arange(0.01, 1, 0.05, dtype=float)
    parameters_grid = {'C': c_parameter,'gamma': gamma_parameter}


    def __init__(self, test_idx, fs_methode, features, labels, score):
        self.test_idx = test_idx
        self.time_start = d.datetime.now()
        self.fs_methode = fs_methode
        self.features = features
        self.labels = labels
        self.score = score
        self.output_path = "Result_Daten\\" + test_idx
        self.best_score, self.best_parameters, self.best_c_gamma, self.best_estimator, self.selected_features = self.tunning_parameter_sfs()
        self.time_end = d.datetime.now()
        self.dauer = (self.time_end - self.time_start)

    def tunning_parameter_sfs(self):
        features = self.features
        labels = self.labels
        score = self.score
        # pre-processing
        features = preprocessing.scale(features)

        best_score = 0
        run_info = pd.DataFrame()
        for C in self.c_parameter:
          for gamma in self.gamma_parameter:

            clf = SVC(kernel="rbf",C=C,gamma=gamma)
            cg_info = pd.DataFrame({"C": [C], "gamma": gamma})
            grid = feature_selection.Sequential_Feature_Selector.sfs_selectors(features, labels, clf, score)
            run_info_temp = pd.concat((cg_info, pd.DataFrame(grid.cv_results_)), axis=1)
            run_info = pd.concat((run_info, run_info_temp), axis=0)
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_parameters = grid.best_params_
                best_c_gamma = [C,gamma]
                best_estimator = grid.best_estimator_
                selected_k_feature = grid.best_estimator_.steps[0][1].k_feature_idx_
        output_name = self.output_path + "\\SVM_"+self.score+"_tunning_parameter_info.csv"
        run_info.to_csv(output_name)
        return best_score, best_parameters, best_c_gamma, best_estimator, selected_k_feature

class RF():


    name = "RF"
    n_parameter = np.arange(1, 100, 1, dtype=int)
    n_parameter = [50,55,60,65,70]
    parameters_grid = {'classifier__n_estimators': n_parameter}

    def __init__(self, test_idx, fs_methode, features, labels, score):
        self.test_idx = test_idx
        self.time_start = d.datetime.now()
        self.fs_methode = fs_methode
        self.features = features
        self.labels = labels
        self.score = score
        self.output_path = "Result_Daten\\" + test_idx
        self.best_score, self.best_parameters_FS, self.best_parameters_clf, self.best_estimator, self.selected_features = self.tunning_parameter_sfs()
        self.time_end = d.datetime.now()
        self.dauer = (self.time_end - self.time_start)
        # self.wirte_info_to_txt()
        # self.wirte_selected_feature_to_csv()

    def tunning_parameter_sfs(self):
        features = self.features
        labels = self.labels

        # pre-processing
        features = preprocessing.scale(features)

        best_score = 0
        run_info = pd.DataFrame()
        loop = 0
        for n in self.n_parameter:
            loop = loop + 1
            print("loop:", loop)
            clf = RandomForestClassifier(n_estimators=n)

            k_info = pd.DataFrame([n], columns=["n"])
            print("RF n= ", n, "start")
            grid = feature_selection.Sequential_Feature_Selector.sfs_selectors(features, labels, clf, self.score)
            run_info_temp = pd.concat((k_info, pd.DataFrame(grid.cv_results_)), axis=1)
            run_info = pd.concat((run_info, run_info_temp), axis=0)

            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_parameters_FS = grid.best_params_
                best_parameters_clf = [n]
                best_estimator = grid.best_estimator_
                selected_k_feature = grid.best_estimator_.steps[0][1].k_feature_idx_
            print("RF n= ", n, "end")
        output_name = self.output_path + "\\" + self.name + "_" + self.fs_methode + "_" + str(
            self.score) + "_tunning_parameter_info.csv"
        run_info.to_csv(output_name)
        return best_score, best_parameters_FS, best_parameters_clf, best_estimator, selected_k_feature


# class SVM():
#     name = "SVM"
#     c_parameter = np.arange(0.1, 10, 0.5, dtype=float)
#     gamma_parameter = np.arange(0.01, 1, 0.05, dtype=float)
#     parameters_grid = {'C': c_parameter, 'gamma': gamma_parameter}
#
#     def __init__(self, test_idx, fs_methode, features, labels, score):
#         self.test_idx = test_idx
#         self.time_start = d.datetime.now()
#         self.fs_methode = fs_methode
#         self.features = features
#         self.labels = labels
#         self.score = score
#         self.output_path = "Result_Daten\\" + test_idx
#         self.best_score, self.best_parameters, self.best_c_gamma, self.best_estimator, self.selected_features = self.tunning_parameter_sfs()
#         self.time_end = d.datetime.now()
#         self.dauer = (self.time_end - self.time_start)
#
#     def tunning_parameter_sfs(self):
#         features = self.features
#         labels = self.labels
#         score = self.score
#         # pre-processing
#         features = preprocessing.scale(features)
#
#         best_score = 0
#         run_info = pd.DataFrame()
#         for C in self.c_parameter:
#             for gamma in self.gamma_parameter:
#
#                 clf = SVC(kernel="rbf", C=C, gamma=gamma)
#                 cg_info = pd.DataFrame({"C": [C], "gamma": gamma})
#                 grid = feature_selection.Sequential_Feature_Selector.sfs_selectors(features, labels, clf, score)
#                 run_info_temp = pd.concat((cg_info, pd.DataFrame(grid.cv_results_)), axis=1)
#                 run_info = pd.concat((run_info, run_info_temp), axis=0)
#                 if grid.best_score_ > best_score:
#                     best_score = grid.best_score_
#                     best_parameters = grid.best_params_
#                     best_c_gamma = [C, gamma]
#                     best_estimator = grid.best_estimator_
#                     selected_k_feature = grid.best_estimator_.steps[0][1].k_feature_idx_
#         output_name = self.output_path + "\\SVM_" + self.score + "_tunning_parameter_info.csv"
#         run_info.to_csv(output_name)
#         return best_score, best_parameters, best_c_gamma, best_estimator, selected_k_feature

class NN():
    name = "NN"

    def model(self,features,labels):
        model = Sequential()
        model.add(Dense(units=112, input_dim=112))

        model.add(Activation('relu'))
        model.add(Dense(units=64, input_dim=64))
        model.add(Activation('relu'))
        model.add(Dense(units=32, input_dim=32))
        model.add(Activation('relu'))
        model.add(Dense(units=17))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])

        print('Training: ')
        # Another way to train the model
        model.fit(features, labels, epochs=10, batch_size=8)

        print('\nTesting: ')
        # Evaluate the model with the metrics we defined earlier
        loss, accuracy = model.evaluate(features, labels)

        # print('test loss: ', loss)
        print('test accuracy: ', accuracy)


class Scroce():
    score = ["accuracy", "average precison"]
    #
    # 'micro':
    # Calculate metrics globally by counting the total true
    # positives,false negatives and false positives.
    #
    # 'macro':
    # Calculate metrics for each label, and find their unweighted mean.
    # This does not take label imbalance into account.
    """
        'weighted':
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label).
        This alters ‘macro’ to account for label imbalance;
        it can result in an F-score that is not between precision and recall.
    """
    f2_score = make_scorer(fbeta_score, beta=2)
    recall_16 = make_scorer(recall_score, labels=[16], average=None)
    recall_macro =make_scorer(recall_score, average="macro")


# path ="Result_Daten\\test\\"+"FE_WP(TestData_1700.csv).csv"
# data =pd.read_csv(path)
# features = data.drop(["label"],axis=1)
# labels = pd.DataFrame(data.ix[:, data.shape[1] - 1]).values.ravel()
# ml_test =KNN(test_idx="test",fs_methode="SFS",features=features,labels=labels,score="accuracy")


