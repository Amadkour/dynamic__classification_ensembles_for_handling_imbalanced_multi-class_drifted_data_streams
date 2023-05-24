import threading
import time

from skmultiflow.drift_detection import DDM, ADWIN

import csm
import numpy as np
import helper as h
from tqdm import tqdm
import multiprocessing
from csm import OOB, UOB, SampleWeightedMetaEstimator, Dumb, MDET, SEA, StratifiedBagging
from strlearn.evaluators import TestThenTrain
from sklearn.naive_bayes import GaussianNB
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity
)
import sys
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.trees import HoeffdingTree, HoeffdingTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Select streams and methods
from experiments.train_and_test_stratigy import MyTestThenTrain

streams = h.realstreams()

ht_0 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU1", )
ht_1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", )
ht_2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU1", )
ht_3 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU1", )
methods=[
    # 'mlsomte',
    # 'mlsol',
    'dynamic',
]

# Define worker
def worker(i, resampling_method):
    stream = streams['covtypeNorm-1-2vsAll']
    drift_methods = [
        ADWIN(),
    ]
    classifiers = [
        (ht_0,ht_0),
    ]

    for index, classifer_name in enumerate(classifiers):
        cclfs = [clone(clf) for clf in classifiers[index]]

        for method in drift_methods:
            print("Starting stream %i/%i" % (i + 1, len(streams)))
            eval = MyTestThenTrain(metrics=(
                balanced_accuracy_score,
                geometric_mean_score_1,
                f1_score,
                precision,
                recall,
                specificity
            ),
            )
            eval.process(
                stream,
                cclfs,
                edit=resampling_method,
                concept_drift_method=method
            )

            print("Done stream %i/%i" % (i + 1, len(streams)))

            results = eval.scores
            np.save("experiments/experiment3/results/ht/%s_%s" % (str(stream).split('/')[1].split('.')[0],resampling_method,), results)


jobs = []
if __name__ == '__main__':
    from joblib import Parallel, delayed

    start_time = time.perf_counter()
    result = Parallel(n_jobs=4, prefer="threads")(delayed(worker)(i, method) for i, method in enumerate(methods))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    print(result)
