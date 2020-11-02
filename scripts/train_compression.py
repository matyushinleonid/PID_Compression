from config import config
from pidc.models.compression.ae import AE
from pidc.utils.data import load_train_val_test


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test(kind='compression')

    model = AE()
    model.fit(X_train, y_train, X_val, y_val)
    model.save()


if __name__ == '__main__':
    main()
