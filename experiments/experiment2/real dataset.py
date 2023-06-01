import time

from skmultiflow.drift_detection import ADWIN

import numpy as np
import helper as h
from csm import SEA, StratifiedBagging
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity
)
from sklearn.base import clone
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.svm import SVC

# Select streams and methods
from csm.train_and_test_stratigy import MyTestThenTrain
streams = h.realstreams()
svm_u1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42)), des="KNORAU1")
svm_e1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42), ), des="KNORAE1")
svm_u2 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42)), des="KNORAU2")
svm_e2 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42), ), des="KNORAE2")

ht_u1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU1")
ht_e1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier(), ), des="KNORAE1")
ht_u2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2")
ht_e2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier(), ), des="KNORAE2")



# Define worker
def worker(i, stream_n):
    stream = streams
    drift_methods = [
        ADWIN(),
    ]

    classifiers = [
        (svm_u1, svm_e1, svm_u2, svm_e2),
        # (ht_u1, ht_e1, ht_u2, ht_e2),
    ]

    cclfs = [clone(clf) for clf in classifiers[0]]
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
        np.save("experiments/experiment2/results_%s" % str(stream).split('/')[1].split('.')[0], results)


jobs = []
if __name__ == '__main__':
    start_time = time.perf_counter()
    worker(0,0)
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
