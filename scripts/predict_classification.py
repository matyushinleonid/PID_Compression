from pidc.models.classification.xgboost import XGBoost
from pidc.utils.data import load_train_val_test


def main():
    xgb = XGBoost.load()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test(kind='classification_on_initial')
    xgb.predict(X_test, save=True, filename_suffix='initial_data_test')

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test(kind='classification_on_reconstructed')
    xgb.predict(X_test, save=True, filename_suffix='reconstructed_test')


if __name__ == '__main__':
    main()
