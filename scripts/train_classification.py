from pidc.models.classification.xgboost import XGBoost
from pidc.utils.data import load_train_val_test

def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()
    xgb = XGBoost(model_name='my_xgboost', n_estimators=10000)
    xgb.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=20)
    xgb.save()

if __name__ == '__main__':
    main()