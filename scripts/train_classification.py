from config import config
from pidc.models.classification.xgboost import XGBoost
from pidc.utils.data import load_train_val_test


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test(kind='classification_on_initial')
    xgb = XGBoost()
    xgb.fit(X_train, y_train, X_val, y_val)
    xgb.save()

if __name__ == '__main__':
    main()
