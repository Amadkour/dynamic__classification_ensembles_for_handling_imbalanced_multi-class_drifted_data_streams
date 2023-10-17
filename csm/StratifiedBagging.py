"""
Stratified Bagging.
"""

from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn import base
from sklearn.metrics import f1_score, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from sklearn.utils.multiclass import unique_labels

ba = balanced_accuracy_score
f1 = f1_score
gmean = geometric_mean_score


class StratifiedBagging(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, ensemble_size=10, random_state=42, oversampler="None"):
        """Initialization."""
        # self._base_clf = base_estimator
        self.ensemble_size = ensemble_size
        self.oversampler = oversampler
        self.estimators_ = []
        self.base_estimator = base_estimator
        self.random_state = random_state

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        # if not hasattr(self, "base_estimator"):
        # self.set_base_clf()
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        minority_X = X[y == 1]
        minority_y = y[y == 1]
        majority_X = X[y == 0]
        majority_y = y[y == 0]

        for i in range(self.ensemble_size):
            self.estimators_.append(base.clone(self.base_estimator))

        for n, estimator in enumerate(self.estimators_):
            np.random.seed(self.random_state + (n * 2))
            bagXminority = minority_X[np.random.choice(minority_X.shape[0], len(minority_y), replace=True), :]
            bagXmajority = majority_X[np.random.choice(majority_X.shape[0], len(majority_y), replace=True), :]

            bagyminority = np.ones(len(minority_y)).astype('int')
            bagymajority = np.zeros(len(majority_y)).astype('int')

            train_X = np.concatenate((bagXmajority, bagXminority))
            train_y = np.concatenate((bagymajority, bagyminority))

            unique, counts = np.unique(train_y, return_counts=True)

            if self.oversampler == "ROS":
                ros = RandomOverSampler(random_state=self.random_state + (n * 2))
                try:
                    train_X, train_y = ros.fit_resample(train_X, train_y)
                except:
                    pass
            elif self.oversampler == "B2":
                b2 = BorderlineSMOTE(random_state=self.random_state + (n * 2), kind='borderline-2')
                try:
                    train_X, train_y = b2.fit_resample(train_X, train_y)
                except:
                    pass
            elif self.oversampler == "RUS":
                rus = RandomUnderSampler(random_state=self.random_state + (n * 2))
                try:
                    train_X, train_y = rus.fit_resample(train_X, train_y)
                    # _, ys_counter = np.unique(train_ys, return_counts=True)

                    # if np.sum(ys_counter) < 9:
                    # rus = RandomUnderSampler(random_state=self.random_state+(n*2), sampling_strategy={0:(9-ys_counter[1]), 1:ys_counter[1]})
                    # train_Xs, train_ys = rus.fit_resample(train_X, train_y)
                    # train_X, train_y = train_Xs, train_ys
                    # else:
                    # train_X, train_y = train_Xs, train_ys
                except:
                    pass
            elif self.oversampler == "CNN":
                cnn = CondensedNearestNeighbour(random_state=self.random_state + (n * 2))
                try:
                    train_X, train_y = cnn.fit_resample(train_X, train_y)
                except:
                    pass
            # if train_X.shape[0] >= 5:
            estimator.fit(train_X, train_y)
            # else:
            #     print("Padlem, więc biorę %i sasiadow" % train_X.shape[0])
            #     self.estimators_[n] = KNeighborsClassifier(weights='distance', n_neighbors=train_X.shape[0]).fit(train_X, train_y)

        # Return the classifier
        return self

    # def partial_fit(self, X, y, classes_ = None, sample_weight=None):
    #     if not hasattr(self, "_fitted"):
    #         # print("ROBIE COS")
    #         self.fit(X, y)
    #         self._fitted=True
    #     for estimator in self.estimators_:
    #         estimator.partial_fit(X, y, classes=classes_, sample_weight=sample_weight)

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.estimators_])

    def predict(self, X):
        """Hard decision."""

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        return prediction

    def predict_proba(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)

        return average_support

    def score(self, X, y):
        return ba(y, self.predict(X))
