import time

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pandas as pd
from preprocessing.mlsol import MLSOL
from preprocessing.mlsmote import MLSMOTE


class MyTestThenTrain:

    def __init__(
            self, metrics=(accuracy_score, balanced_accuracy_score), verbose=False
    ):
        self.balanced_methods_details = []
        self.balanced_methods = ["MLSmote", "MLSOL", "adaptive"]
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]
        self.verbose = verbose
        self.chunk_length = 200
        # Prepare scores table
        self.scores = np.zeros(
            (len(self.balanced_methods), self.chunk_length, len(self.metrics))
        )
        self.execution_time = np.zeros(len(self.balanced_methods))
        self.overlappedItems = np.zeros((len(self.balanced_methods), self.chunk_length)
                                        )

    def process(self, stream, clfs, concept_drift_method=None, imbalance_method='mlsmote'):
        # Verify if pool of classifiers or one
        if isinstance(clfs, ClassifierMixin):
            self.clfs_ = [clfs]
        else:
            self.clfs_ = clfs

        # Assign parameters
        self.stream_ = stream

        index = 0
        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
        imbalance_pool = []
        start_time = time.perf_counter()
        while True:
            chunk = stream.get_chunk()

            index += 1
            X, y = chunk
            print(index)

            if self.verbose:
                pbar.update(1)
            # Test
            if stream.previous_chunk is not None:

                # prediction
                for clfid, clf in enumerate(self.clfs_):
                    y_pred = clf.predict(X)

            if stream.previous_chunk is not None and concept_drift_method is not None:
                '''concept drift'''
                for i, score in enumerate(y_pred):
                    in_drift, in_warning = concept_drift_method.update(score)
                    if in_drift:
                        imbalance_pool.append(y[index])
                if len(imbalance_pool) > 0:
                    # concept_drift_method.reset()
                    minority_classes = self.myProposedRation(y)
                    if len(minority_classes) > 0:
                        unique_elements, counts = np.unique(y, return_counts=True)
                        for minority in minority_classes:
                            X, y = self.balanceding(X, y, minority, stream.chunk_id, imbalance_method)
                        unique_elements, counts = np.unique(y, return_counts=True)

                    # Train

                    [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]

            else:
                # Train
                [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]
                # accuracy
            predictions = [[0 for _ in range(len(y))] for _ in self.metrics]
            for clfid, clf in enumerate(self.clfs_):
                predictions[clfid] = clf.predict(X)
            mean_prediction = np.mean(predictions, axis=0)

            self.scores[self.balanced_methods.index(imbalance_method), stream.chunk_id - 1] = [
                metric(y, mean_prediction) for metric in self.metrics]
            if stream.is_dry():
                break
        np.save("results-ddm/sub_score_" + imbalance_method, self.scores)
        np.save("results-ddm/sub_overlapped_score_" + imbalance_method, self.overlappedItems)
        np.save("results-ddm/sub_time_score_" + imbalance_method, self.execution_time)
        finish_time = time.perf_counter()
        self.execution_time[self.balanced_methods.index(imbalance_method)] = finish_time - start_time

    def myProposedRation(self, labels):
        unique_elements, counts = np.unique(labels, return_counts=True)
        best_frq = np.mean(counts)
        minority_classes = []
        for i, frq in enumerate(counts):
            if frq < (best_frq * 0.25):
                minority_classes.append(unique_elements[i])

        return minority_classes

    synthetic_deviation = 0.1

    def balanceding(self, X, y, minority_classes, stream_id, edit=None, scal=0.1):
        indices_of_minority_classes = [i for i, element in enumerate(y) if element == minority_classes]
        minority_instances = [X[i] for i in indices_of_minority_classes]
        minority_label = [y[i] for i in indices_of_minority_classes]
        unique_elements, counts = np.unique(y, return_counts=True)
        if len(minority_instances):
            minority_instances.append(minority_instances[0])
            minority_instances.append(minority_instances[0])
            minority_label.append(minority_label[0])
            minority_label.append(minority_label[0])
        best_frq = int(np.mean(counts))
        if edit == self.balanced_methods[1]:
            minority_instances, minority_label = MLSOL(self.synthetic_deviation).fit_resample(minority_instances,
                                                                                              minority_label)
        elif edit == self.balanced_methods[0]:
            data = pd.DataFrame(minority_instances)

            label = pd.get_dummies(minority_label, prefix='class')
            data, label = MLSMOTE(data, label, best_frq)
            minority_instances = data.to_numpy()
            minority_label = np.array(label)
        elif edit == self.balanced_methods[2]:
            distance = 10000000
            method = 'mlsmote'
            current_mean = np.mean(X)
            current_std = np.std(X)
            # print('current mean and std', current_mean, ' and ', current_std)
            for index, value in enumerate(self.balanced_methods_details):
                meth, mean, std = value
                dist = np.sqrt(np.square(current_mean - mean) + np.square(current_std - std))
                # print('distance of (', value, ') is equal :', dist)
                if dist < distance:
                    distance = dist
                    method = meth

            if method == self.balanced_methods[0]:
                # -----mlsol------------#

                self.balanced_methods_details.append((self.balanced_methods[1], np.mean(X), np.std(X)))
                minority_instances, minority_label = MLSOL(scal).fit_resample(minority_instances, minority_label)
            else:
                # -----mlsmote------------#
                self.balanced_methods_details.append((self.balanced_methods[0], np.mean(X), np.std(X)))
                data = pd.DataFrame(minority_instances)
                label = pd.get_dummies(minority_label, prefix='class')
                data, label = MLSMOTE(data, label, best_frq)
                minority_instances = data.to_numpy()
                minority_label = np.array(label)

            # print('current chunk use: ', method)

        else:
            pass
        new_X = np.concatenate([X, minority_instances], axis=0)
        new_y = np.concatenate([y, minority_label], axis=0)
        if edit != self.balanced_methods[2]:
            # check overlapped
            self.check_overlapped(X, y, minority_instances, minority_label, stream_id, edit)
            return new_X, new_y
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(new_X, new_y)
        s_x = []
        s_y = []
        _, indices = knn.kneighbors(minority_instances)
        for i, index in enumerate(indices):
            neighbour = np.unique(new_y[index])
            if len(neighbour) == 1 and neighbour[0] == minority_classes:
                s_x.append(minority_instances[i])
                s_y.append(minority_label[i])
        if len(s_x) == 0:
            self.synthetic_deviation += 0.1

            return self.balanceding(X, y, minority_classes, stream_id, edit, scal=self.synthetic_deviation)
        # check overlapped
        self.check_overlapped(X, y, s_x, s_y, stream_id, edit)
        self.synthetic_deviation = 0.1
        return np.concatenate([X, s_x], axis=0), np.concatenate([y, s_y], axis=0)

    def check_overlapped(self, X, y, new_X, new_y, stream_id, imbalance_method):
        number_of_overlapped = []
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(X, y)
        _, indices = knn.kneighbors(new_X)
        for i, index in enumerate(indices):
            neighbour = np.unique(y[index])
            if len(neighbour) > 1:
                number_of_overlapped.append(new_X[i])
        self.overlappedItems[self.balanced_methods.index(imbalance_method), stream_id - 1] = len(number_of_overlapped)
    # Define a function to calculate IRLbl for a given label 'l'
