from pidc.utils.data import split_train_val_test


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_train_val_test(save=True)

if __name__ == '__main__':
    main()