from .base import BaseClassifier
from xgboost import XGBClassifier
import pandas as pd
import os
from config import config
import pickle


class XGBoost(BaseClassifier):

    def __init__(self, model_name='xgboost', **kwargs):
        self._model = XGBClassifier(**kwargs)
        self.model_name = model_name

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        self._model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **kwargs)

    def predict(self, X, **kwargs):
        preds = self._model.predict(X, **kwargs)

        return pd.DataFrame(preds, columns=[config['classification']['target_column']])

    def save(self):
        model_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
        if not os.path.isdir(model_dir_path):
            os.mkdir(model_dir_path)

        with open(model_dir_path / 'model.pickle', 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_name):
        model_dir_path = config['data_dir'] / 'cache' / 'models' / model_name
        with open(model_dir_path / 'model.pickle', 'rb') as f:
            self = pickle.load(f)

        return self


