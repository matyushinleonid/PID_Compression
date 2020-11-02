from pidc.models.compression.ae import AE
from pidc.utils.data import load_train_val_test


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test(kind='compression')

    model = AE.load()
    for name, df in {'X_train': X_train, 'X_val': X_val, 'X_test': X_test}.items():
        model.predict(df, save=True, filename_suffix=f'{name}')


if __name__ == '__main__':
    main()
