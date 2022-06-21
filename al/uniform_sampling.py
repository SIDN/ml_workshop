import pandas as pd

from . import SamplingMethod


class UniformSampling(SamplingMethod):
    """
    Sample random data points, i.e. giving each data point a fair chance of being sampled.

    """

    @staticmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, **kwargs) -> list:
        """
        Returns batch of data points using uniform sampling.

        :param pool: Dataframe with data points that can be sampled.
        :param nr_samples: Number of data points that should be sampled.
        :param kwargs: Additional arguments.
        :return: Indices of sampled data points.
        """
        return pool.sample(n=nr_samples, random_state=kwargs.get('seed', None)).index
