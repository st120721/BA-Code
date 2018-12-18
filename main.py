import os
import datetime as d

from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split

from ml_algoritms import ML_Algoritms,Scroce
from feature_extraction import Feature_Extraction


def main():
    raw_data = "TestData_1700.csv"
    recall_macro = make_scorer(recall_score, average="macro")
    for score in["accuracy"]:

        # for clf in["KNN","SVM","RF"]:
        for clf in ["RF"]:

            test_idx=clf+"_"+str(score)+"_"+ str(d.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
            fe=Feature_Extraction(test_idx=test_idx, data_name=raw_data,
                                transformation_name="WP",transformation_level =4)
            features=fe.features
            labels =fe.labels.values.ravel()

            # train test split
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

            ML_Algoritms(classifier=clf,test_idx=test_idx,
                              fs_methode="SFS", features=X_train, labels=y_train, score=score)

if __name__ == "__main__":
    main()