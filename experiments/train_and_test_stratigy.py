import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
import pandas as pd
from preprocessing.mlsol import MLSOL
from preprocessing.mlsmote import MLSMOTE


class MyTestThenTrain:
    """
    Test Than Train data stream evaluator.

    Implementation of test-then-train evaluation procedure,
    where each individual data chunk is first used to test
    the classifier and then it is used for training.

    :type metrics: tuple or function
    :param metrics: Tuple of metric functions or single metric function.
    :type verbose: boolean
    :param verbose: Flag to turn on verbose mode.

    :var classes_: The class labels.
    :var scores_: Values of metrics for each processed data chunk.
    :vartype classes_: array-like, shape (n_classes, )
    :vartype scores_: array-like, shape (stream.n_chunks, len(metrics))

    :Example:

    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.classifiers.AccumulatedSamplesClassifier()
    >>> evaluator = sl.evaluators.TestThenTrainEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    ...
    [[0.92       0.91879699 0.91848191 0.91879699 0.92523364]
    [0.945      0.94648779 0.94624912 0.94648779 0.94240838]
    [0.92       0.91936979 0.91936231 0.91936979 0.9047619 ]
    ...
    [0.92       0.91907051 0.91877671 0.91907051 0.9245283 ]
    [0.885      0.8854889  0.88546135 0.8854889  0.87830688]
    [0.935      0.93569212 0.93540766 0.93569212 0.93467337]]
    """

    def __init__(
            self, metrics=(accuracy_score, balanced_accuracy_score), verbose=False
    ):
        self.balanced_methods = []
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]
        self.verbose = verbose

    def process(self, stream, clfs, concept_drift_method=None, edit=None):
        """
        Perform learning procedure on data stream.

        :param stream: Data stream as an object
        :type stream: object
        :param clfs: scikit-learn estimator of list of scikit-learn estimators.
        :type clfs: tuple or function

        Parameters
        ----------
        use_concept_drift
        """
        # Verify if pool of classifiers or one
        if isinstance(clfs, ClassifierMixin):
            self.clfs_ = [clfs]
        else:
            self.clfs_ = clfs

        # Assign parameters
        self.stream_ = stream

        # Prepare scores table
        self.scores = np.zeros(
            (len(self.clfs_), ((self.stream_.n_chunks - 1)), len(self.metrics))
        )
        index = 0
        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
        have_drift = 0
        while True:
            chunk = stream.get_chunk()
            index += 1
            X, y = chunk

            if self.verbose:
                pbar.update(1)

            # Test
            y_prediction = np.zeros(y.shape)
            if stream.previous_chunk is not None:
                for clfid, clf in enumerate(self.clfs_):
                    y_pred = clf.predict(X)
                    for i in range(len(y_prediction)):
                        y_prediction[i] = y_prediction[i] + y_pred[i]

                    self.scores[clfid, stream.chunk_id - 1] = [
                        metric(y, y_pred) for metric in self.metrics]
            if stream.previous_chunk is not None and concept_drift_method is not None:
                '''concept drift'''
                for score in y_prediction:
                    concept_drift_method.add_element(score)
                    if concept_drift_method.detected_change():
                        have_drift += 1
                print(index, 'have  score : ', have_drift)
                if have_drift > 0:
                    concept_drift_method.reset()
                    have_drift = 0
                    print('Change detected at chunk number', index)
                    self.balanceding(X, y, edit)
                    # Train
                    [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]
            else:
                print(index)
                # Train
                [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]
                print(index, ' is finish')

            if stream.is_dry():
                break

    def balanceding(self, X, y, edit=None):
        if edit == 'mlsol':
            X, y = MLSOL().fit_resample(X, y)
        elif edit == 'mlsmote':
            data = pd.DataFrame(X)
            label = pd.get_dummies(y, prefix='class')
            data, label = MLSMOTE(data, label, 50)
            X = data.to_numpy()
            y = np.array(label)
        elif edit == 'dynamic':
            distance = 10000000
            method = 'mlsmote'
            current_mean = np.mean(X)
            current_std = np.std(X)
            print('current mean and std', current_mean, ' and ', current_std)
            for index, value in enumerate(self.balanced_methods):
                meth, mean, std = value
                dist = np.sqrt(np.square(current_mean - mean) + np.square(current_std - std))
                print('distance of (', value, ') is equal :', dist)
                if dist < distance:
                    distance = dist
                    method = meth

            if method == 'mlsmote':
                # -----mlsol------------#

                self.balanced_methods.append(('mlsol', np.mean(X), np.std(X)))
                X, y = MLSOL().fit_resample(X, y)
            else:
                # -----mlsmote------------#
                self.balanced_methods.append(('mlsmote', np.mean(X), np.std(X)))
                data = pd.DataFrame(X)
                label = pd.get_dummies(y, prefix='class')
                data, label = MLSMOTE(data, label, 50)
                X = data.to_numpy()
                y = np.array(label)

            print('current chunk use: ', method)

        else:
            pass
        return X, y
