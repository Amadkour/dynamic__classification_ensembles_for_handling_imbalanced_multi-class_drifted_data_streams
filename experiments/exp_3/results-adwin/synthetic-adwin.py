import time
import warnings

from river import drift
import numpy as np
import helper as h
from csm import SEA, StratifiedBagging
from sklearn.naive_bayes import GaussianNB
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity
)
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Select streams and methods
from csm.train_and_test_stratigy import MyTestThenTrain

warnings.filterwarnings("ignore")
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
#
# ht_u1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU1",oversampled='None')
# ht_e1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier(), ), des="KNORAE1",oversampled='dynamic')
# ht_u2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2",oversampled="mlsmote")
# ht_e2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier(), ), des="KNORAE2", oversampled="mlsol")
#
# ht_0 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled=None)
# ht_1 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled='dynamic')
# ht_2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled="mlsmote")
# ht_3 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="KNORAU2", oversampled="mlsol")

comcept_method = drift.ADWIN()
alternative_preprocessing_method = ["MLSmote", "MLSOL", "adaptive"]


# Define worker
def worker():
    classifiers_name = [
        # 'svm',
        #                 'knn',
        #                 'gnb',
        'mix'
    ]
    classifiers = [
        # (svm_u1, svm_e1, svm_u2, svm_e2),
        # (knn_u1,knn_u1, knn_u2, knn_u2),
        (knn_u1, svm_u1, gnb_u1, gnb_u1),
    ]

    for index, classifer_name in enumerate(classifiers_name):
        cclfs = [clone(clf) for clf in classifiers[index]]
        eval = MyTestThenTrain(metrics=(
            balanced_accuracy_score,
            geometric_mean_score_1,
            f1_score,
            precision,
            recall,
        ))
        for method in alternative_preprocessing_method:
            streams = h.synthetic_streams(123)
            stream = streams[list(streams.keys())[0]]
            eval.process(
                stream,
                cclfs,
                concept_drift_method=comcept_method,
                imbalance_method =method
            )

        np.save("results-adwin/score", eval.scores)
        np.save("results-adwin/score_overlapped", eval.overlappedItems)
        np.save("results-adwin/score_time", eval.execution_time)
        # np.save("experiments/experiment2/results-adwin/svm/DDM/%s" % (str(stream).split('/')[1].split('.')[0]),results-adwin)


jobs = []
if __name__ == '__main__':
    start_time = time.perf_counter()

    worker()
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    # from joblib import Parallel, delayed
    #
    # start_time = time.perf_counter()
    # result = Parallel(n_jobs=4, prefer="threads")(delayed(worker)(i, stream_n) for i, stream_n in enumerate(streams))
    # finish_time = time.perf_counter()
    # print(f"Program finished in {finish_time - start_time} seconds")
    # print(result)
