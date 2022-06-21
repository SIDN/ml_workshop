import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from . import SamplingMethod


class UncertaintySampling(SamplingMethod):
    """
    Sample data points a model is uncertain about, i.e. data points close a model's decision boundary.

    See also: https://medium.com/@hardik.dave/active-learning-sampling-strategies-f8d8ac7037c8#cca6

    """

    @staticmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, model: BaseEstimator, **kwargs) -> list:
        """
        Returns batch of data points using uncertainty sampling.

        :param pool: Dataframe with data points that can be sampled.
        :param nr_samples: Number of data points that should be sampled.
        :param model: A fitted sklearn estimator that is used to compute uncertainty.
        :param kwargs: Additional arguments.
        :return: Indices of sampled data points.
        """
        try:
            distances = model.decision_function(pool)
        except AttributeError:
            distances = model.predict_proba(pool)

        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        active_samples = rank_ind[:nr_samples]
        return pool.iloc[active_samples].index
