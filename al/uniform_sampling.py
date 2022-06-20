import pandas as pd

from . import SamplingMethod


class UniformSampling(SamplingMethod):
    @staticmethod
    def select_batch(pool: pd.DataFrame, nr_samples: int, **kwargs) -> list:
        return pool.sample(n=nr_samples, random_state=kwargs.get('seed', None)).index
