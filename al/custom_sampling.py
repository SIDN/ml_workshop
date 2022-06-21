import numpy as np
import pandas as pd
from typing import List
from scipy.stats import entropy
from sklearn.base import BaseEstimator

from . import SamplingMethod


class CustomSampling(SamplingMethod):
    """
    Your custom sampling strategy
    """

    @staticmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, model: BaseEstimator, committee: List[BaseEstimator], **kwargs) -> list:
        """
        Returns batch of data points using your sampling method.

        :param pool: Dataframe with data points that can be sampled.
        :param nr_samples: Number of data points that should be sampled.
        :param model: A fitted sklearn estimator that is used to compute uncertainty. (You might not use this)
        :param committee: A voting committee of at least two fitted sklearn estimators. (You might not use this)
        :param kwargs: Additional arguments.
        :return: Indices of sampled data points.
        """
        return []
