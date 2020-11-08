from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def fit(self, cond_train, data_train, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate(self, cond, save=True, filename_suffix=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls):
        raise NotImplementedError
