# Inspired by: https://github.com/google/active-learning
# Inspired by: https://modal-python.readthedocs.io/en/latest/index.html
from abc import abstractmethod, ABC
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import cross_validate


def run_iteration(iteration: int, clf: BaseEstimator, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                  y_test: pd.Series):
    cv_scores = cross_validate(clf, x_train, y_train, cv=5, scoring=['average_precision', 'precision', 'recall'])
    cv_scores_avg = pd.DataFrame(cv_scores).mean()
    cv_scores_std = pd.DataFrame(cv_scores).std()

    clf = clone(clf)  # construct a new unfitted estimator with the same parameters.
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]

    return {'iteration': iteration,
            'num_samples': x_train.shape[0],
            'num_pos': (y_train == 1).sum(),
            'num_neg': (y_train == 0).sum(),
            'cv_ap': cv_scores_avg.test_average_precision,
            'cv_precision': cv_scores_avg.test_precision,
            'cv_recall': cv_scores_avg.test_recall,
            'cv_ap_std': cv_scores_std.test_average_precision,
            'cv_precision_std': cv_scores_std.test_precision,
            'cv_recall_std': cv_scores_std.test_recall,
            'test_ap': average_precision_score(y_test, y_prob),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred),
            'test_pr_curve': precision_recall_curve(y_test, y_prob)}


def plot_iterations(df_results: pd.DataFrame, metric: str, ax: Union[plt.axis, None] = None, max_iteration: Union[int, None] = None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.grid()

    if metric == 'num_samples':
        ax.plot(df_results.iteration, df_results['num_pos'], "o-", color="C0", label="Number of positive samples")
        ax.plot(df_results.iteration, df_results['num_neg'], "o-", color="C1", label="Number of negative samples")
    elif metric in ['ap', 'precision', 'recall']:
        ax.errorbar(df_results.iteration, df_results[f'cv_{metric}'], yerr=df_results[f'cv_{metric}_std'], color="C0",
                    label="Cross-validation")
        ax.plot(df_results.iteration, df_results[f'test_{metric}'], "o-", color="C1", label="Test")
        ax.set_ylim([0, 1])
        if max_iteration is not None:
            ax.set_xlim([0, max_iteration])

    ax.set_title(metric)
    ax.legend(loc='best')


class SamplingMethod(ABC):
    """
    Abstract class for sampling methods.

    Provides interface to sampling methods that allow same signature for select_batch. Each subclass implements
    `select_batch` with the desired signature for readability.
    """

    @abstractmethod
    def __init__(self, df_x: pd.DataFrame, y: pd.Series, seed):
        self.df_x = df_x
        self.y = y
        self.seed = seed

    @staticmethod
    @abstractmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, **kwargs) -> list:
        """
        Returns batch of sampled data points.

        :param already_selected:
        :param num_samples:
        :param args:
        :param kwargs:
        :return:
        """
