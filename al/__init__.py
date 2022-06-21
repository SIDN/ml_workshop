from abc import abstractmethod, ABC

import pandas as pd


class SamplingMethod(ABC):
    """
    Abstract class for sampling methods.

    Provides an interface to sampling methods that allow the same signature for select_batch.
    """

    @staticmethod
    @abstractmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, **kwargs) -> list:
        """
        Returns batch of sampled data points.

        :param pool: Dataframe with data points that can be sampled.
        :param nr_samples: Number of data points that should be sampled.
        :param kwargs: Additional arguments required by a specific sampling method.
        :return: Indices of sampled data points.
        """
