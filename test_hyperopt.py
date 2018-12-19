# now with scaling as an option
import pandas as pd
import  numpy as np
from hyperopt import fmin, Trials, STATUS_OK, hp, tpe
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import datasets, preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import csv
import pandas as pd
# File to save first results

with open('trials.csv',"x"and "a")as f:

    writer = csv.writer(f)

    # Write the headers to the file
    writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])



iris = datasets.load_iris()
X = iris.data
y = iris.target


# pre-processing
features = preprocessing.scale(X)
best_score = 0
run_info = pd.DataFrame()
loop = 0
sfs_param_space = {"k_features": hp.choice("k_features",np.arange(1, 3, 1, dtype=int).tolist()),
                   "forward":  hp.choice("forward",[True, False]),
                   "floating":  hp.choice( "floating",[True, False]),
                   }

knn_param_space = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 50)),


}
parm_space = dict(sfs_param_space,**knn_param_space)

# parm_space={'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
#             "k_features": hp.choice("k_features",np.arange(1,3,1,dtype=int).tolist())
#             }

print(parm_space)
loop = 0

best_sfs_idx=[]
best_score =0
best_k=0
best_n=0
def hyperopt_train_test(params):
    #
    n_neighbors = params['n_neighbors']
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)


    #
    k_features =params["k_features"]
    foward = params["forward"]
    floating = params["floating"]

    if foward ==True and floating ==False:
        sfs_="SFS"
    elif foward ==False and floating ==False:
        sfs_="SBS"
    elif foward == True and floating ==True:
        sfs_="FSFS"
    elif foward == False and floating ==True:
        sfs_ ="FBFS"


    global loop
    loop =loop+1
    print(loop)
    print("start tunning param")
    print("k=", k_features)
    print("n=",n_neighbors)
    print(floating)
    print(foward)

    sfs = SequentialFeatureSelector(estimator=clf, k_features=k_features,
                                    forward=foward,floating=floating,scoring='accuracy',
                                    n_jobs=1,
                                    cv=5, )
    sfs.fit(X, y)

    # of_connection = open(out_file, 'a')
    with open('trials.csv', "a")as f:


        writer = csv.writer(f)
        writer.writerow([foward, floating, list(sfs.k_feature_idx_), params['n_neighbors']])

    score = sfs.k_score_
    print(sfs.k_feature_idx_.__class__)
    print(score)



    return score,sfs.k_feature_idx_,k_features,n_neighbors

def score(params):
    print(params)
    global best_sfs_idx, best_k, best_n,best_score
    acc,idx,k,n = hyperopt_train_test(params)
    if best_score < acc:

        best_score =acc
        best_sfs_idx=idx
        best_k =k
        best_n =n
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(score, parm_space, algo=tpe.suggest,
            max_evals = 10,
             trials=trials)



print(best_score)
print(best_sfs_idx)
print(best_n)
print(best_k)
print(best)
print(trials.results)
print(trials.best_trial)
