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

svm_u1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42)), des="KNORAU1")
svm_e1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42), ), des="KNORAE1")
svm_u2 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42)), des="KNORAU2")
svm_e2 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42), ), des="KNORAE2")

knn_u1 = SEA(base_estimator=StratifiedBagging(base_estimator=KNeighborsClassifier()), des="KNORAU1")
knn_e1 = SEA(base_estimator=StratifiedBagging(base_estimator=KNeighborsClassifier(), ), des="KNORAE1")
knn_u2 = SEA(base_estimator=StratifiedBagging(base_estimator=KNeighborsClassifier()), des="KNORAU2")
knn_e2 = SEA(base_estimator=StratifiedBagging(base_estimator=KNeighborsClassifier(), ), des="KNORAE2")

gnb_u1 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB()), des="KNORAU1")
gnb_e1 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(), ), des="KNORAE1")
gnb_u2 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB()), des="KNORAU2")
gnb_e2 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(), ), des="KNORAE2")

ht_u1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU1",oversampled='None')
ht_e1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier(), ), des="KNORAE1",oversampled='dynamic')
ht_u2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2",oversampled="mlsmote")
ht_e2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier(), ), des="KNORAE2", oversampled="mlsol")

ht_0 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled=None)
ht_1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled='dynamic')
ht_2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled="mlsmote")
ht_3 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled="mlsol")


# clfs = (
#     svm_u1, svm_e1, svm_u2, svm_e2,
#     #     knn_u1,knn_u1, knn_u2, knn_u2,
#     #     gnb_u1, gnb_e1, gnb_u2, gnb_e2,
#     # ht_u1, ht_u2, ht_e1, ht_e2---->
# )


# Define worker
def worker(i, stream_n):
    stream = streams[stream_n]
    drift_methods = [
        # ADWIN(),
                     DDM()
    ]
    classifiers_name = [
        # 'svm',
        #                 'knn',
        #                 'gnb',
        'ht'
    ]
    classifiers = [
        # (svm_u1, svm_e1, svm_u2, svm_e2),
        # (knn_u1,knn_u1, knn_u2, knn_u2),
        (ht_u1, ht_e1, ht_u2, ht_e2),
    ]

    for index, classifer_name in enumerate(classifiers_name):
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
            ))
            eval.process(
                stream,
                cclfs,
                concept_drift_method=method
            )

            print("Done stream %i/%i" % (i + 1, len(streams)))

            results = eval.scores
            np.save("experiments/experiment2/results/ht/DDM/%s" % (str(stream).split('/')[1].split('.')[0]),results)
            # np.save("experiments/experiment2/results/svm/DDM/%s" % (str(stream).split('/')[1].split('.')[0]),results)


jobs = []
if __name__ == '__main__':
    from joblib import Parallel, delayed

    start_time = time.perf_counter()
    result = Parallel(n_jobs=5, prefer="threads")(delayed(worker)(i, stream_n) for i, stream_n in enumerate(streams))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    print(result)
