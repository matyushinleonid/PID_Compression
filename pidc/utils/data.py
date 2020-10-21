from config import config
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_train_val_test(save=True):
    df = pd.read_csv(config['data_dir'] / 'input' / 'data.csv')

    X, y = df[config['classification']['X_columns']], df[config['classification']['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['classification']['test_size'],
                                                        random_state=config['classification']['random_state'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=config['classification']['val_size'] / (
                                                                  1 - config['classification']['test_size']),
                                                      random_state=1)

    if save:
        for file_name, df in {'X_train': X_train, 'X_test': X_test, 'X_val': X_val, 'y_train': y_train,
                              'y_test': y_test, 'y_val': y_val}.items():
            df.to_csv(config['data_dir'] / 'cache' / 'initial_data_split' / (file_name + '.csv'), index=False)

    [X_train, y_train, X_val, y_val, X_test, y_test] = list(
        map(lambda t: t.reset_index(drop=True), [X_train, y_train, X_val, y_val, X_test, y_test])
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_train_val_test():
    dfs = []
    for file_name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
        dfs.append(pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / (file_name + '.csv')))

    return (dfs[0], dfs[1]), (dfs[2], dfs[3]), (dfs[4], dfs[5])