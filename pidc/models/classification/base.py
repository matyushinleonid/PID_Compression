from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    name = None

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
    def load(cls, model_name):
        raise NotImplementedError