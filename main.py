import os
import datetime as d
from ml_algoritms import ML_Algoritms
from feature_extraction import Feature_Extraction


def main():
    raw_data = "TestData_1700.csv"
    for score in["accuracy","average_precision"]:
        # for clf in["KNN","SVM","RF"]:
        for clf in ["KNN"]:

            test_name=clf+"_"+score+"_"+ str(d.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
            fe=Feature_Extraction(test_name=test_name, data_name=raw_data,
                                transformation_name="WP",transformation_level =4)
            features=fe.features
            labels =fe.labels.values.ravel()
            ML_Algoritms(classifier=clf,test_name=test_name,
                              fs_methode="SFS", features=features, labels=labels, score=score)

if __name__ == "__main__":
    main()