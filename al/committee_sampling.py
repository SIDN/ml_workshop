import numpy as np
import pandas as pd
from typing import List
from scipy.stats import entropy
from sklearn.base import BaseEstimator

from . import SamplingMethod


class CommitteeDisagreementSampling(SamplingMethod):
    """
    Sample data points a committee judges differently, i.e. data points where disagreement occurs.

    See also: https://medium.com/@hardik.dave/active-learning-sampling-strategies-f8d8ac7037c8#07a8

    """

    @staticmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, committee: List[BaseEstimator], **kwargs) -> list:
        """
        Returns batch of data points using committee disagreement sampling.

        :param pool: Dataframe with data points that can be sampled.
        :param nr_samples: Number of data points that should be sampled.
        :param committee: A voting committee of at least two fitted sklearn estimators.
        :param kwargs: Additional arguments.
        :return: Indices of sampled data points.
        """

        vote_proba = []
        for member in committee:
            vote_proba.append(member.predict_proba(pool))
        vote_proba = np.array(vote_proba)

        consensus_proba = np.mean(vote_proba, axis=0)

        learner_KL_div = np.empty_like(consensus_proba)
        for i in range(len(consensus_proba)):
            for j in range(consensus_proba.shape[1]):
                learner_KL_div[i, j] = entropy(vote_proba[j, i], qk=consensus_proba[i])

        max_disagreement = pd.Series(np.max(learner_KL_div, axis=1), index=pool.index)
        return max_disagreement.sort_values(ascending=False)[:nr_samples].index
