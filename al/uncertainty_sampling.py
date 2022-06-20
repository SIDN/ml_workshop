import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from . import SamplingMethod


class UncertaintySampling(SamplingMethod):
    @staticmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, model: BaseEstimator, **kwargs) -> list:
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
