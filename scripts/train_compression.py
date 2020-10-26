from pidc.models.compression.ae import AE
from pidc.utils.data import load_train_val_test
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pandas as pd
from config import config


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    for scaler in [RobustScaler(), MinMaxScaler()]:
        scaler.fit(X_train)
        X_train, X_val, X_test = list(
            map(lambda t: pd.DataFrame(scaler.transform(t), columns=t.columns), [X_train, X_val, X_test])
        )

    model = AE(input_dim=X_train.shape[1], latent_dim=config['compression']['latent_dim'], model_name='my_ae')
    model.fit(X_train, y_train, X_val, y_val)
    model.save()

if __name__ == '__main__':
    main()
