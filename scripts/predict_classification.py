from pidc.models.classification.xgboost import XGBoost
from pidc.utils.data import load_train_val_test

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()
    xgb = XGBoost.load(model_name='my_xgboost')
    xgb.predict(X_test, save=True, filename_suffix='initial_data_test')

if __name__ == '__main__':
    main()