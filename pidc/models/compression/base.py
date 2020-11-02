from abc import ABC, abstractmethod


class BaseCompressor(ABC):
    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, save=True, filename_suffix=None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls):
        raise NotImplementedError
