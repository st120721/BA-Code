from hyperopt import hp
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
KNN = dict(
    name="KNN",
    parameters_list=["n_neighbors"],
    parameters_grid={'n_neighbors': hp.choice('n_neighbors', np.arange(1, 10, 1, dtype=int).tolist())},
    estimator=KNeighborsClassifier(),

)
print(KNN)