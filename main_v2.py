import os
import datetime as d
import pandas as pd
from keras.utils import np_utils
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn import datasets
from machine_learning import Machine_Learning,Scoring
from feature_extraction_wavelet import Feature_Extraction_Wavelet
from feature_extraction_rms import Feature_Extraction_RMS
'''
        
'''
def main():
    raw_data = "TestData_1700.csv"
    for fe_type in ["wavelet"]:
        for clf in ["KNN"]:
            folder_path = "Result_Daten\\"+clf + "_"+fe_type+"_" + str(d.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
            isExists = os.path.exists(folder_path)
            if not isExists:
                os.makedirs(folder_path)

            if fe_type == "wavelet":
                fe = Feature_Extraction_Wavelet(folder_path=folder_path, data_name=raw_data,
                                        transformation_name="WP", transformation_level=4)
            if fe_type == "rms":
                fe =Feature_Extraction_RMS(folder_path=folder_path, data_name=raw_data)
            scorings_dict =Scoring().make_score()

            for scoring_name,scoring_function in scorings_dict.items():

                folder_path_temp=folder_path+"\\"+clf+"_"+str(scoring_name)
                isExists = os.path.exists(folder_path_temp)
                if not isExists:
                    os.makedirs(folder_path_temp)


                features=fe.features.values
                labels =fe.labels.values.ravel()


                # # test
                # iris = datasets.load_iris()
                # features=pd.DataFrame(iris.data).values
                # labels = iris.target


                skf = StratifiedShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7)
                print(skf.split(features, labels))
                for train_index, test_index in skf.split(features, labels):
                    X_train, X_test = features[train_index], features[test_index]
                    y_train, y_test = labels[train_index], labels[test_index]

                if (clf == "DNN"):
                    y_train =y_train-1
                    y_test =y_test-1
                    y_train = np_utils.to_categorical(y_train, num_classes=17)
                    y_test = np_utils.to_categorical(y_train, num_classes=17)

                scoring =[scoring_name,scoring_function]
                Machine_Learning(algorithm=clf,scoring=scoring,loops=20,folder_path=folder_path_temp,
                                   features=X_train, labels=y_train)

if __name__ == "__main__":
    main()