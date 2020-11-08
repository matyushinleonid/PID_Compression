from config import config
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelBinarizer


def split_train_val_test(save=True):
    df = pd.read_csv(config['data_dir'] / 'input' / 'data.csv')

    X, y = df[config['data_import']['X_columns']], df[config['data_import']['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data_import']['test_size'],
                                                        random_state=config['data_import']['random_state'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=config['data_import']['val_size'] / (
                                                                  1 - config['data_import']['test_size']),
                                                      random_state=config['data_import']['random_state'])

    if config['data_import']['rescale'] == True:
        for scaler in [RobustScaler(), MinMaxScaler()]:
            scaler.fit(X_train)
            X_train, X_val, X_test = list(
                map(lambda t: pd.DataFrame(scaler.transform(t), columns=t.columns), [X_train, X_val, X_test])
            )

    if save:
        for file_name, df in {'X_train': X_train, 'X_test': X_test, 'X_val': X_val, 'y_train': y_train,
                              'y_test': y_test, 'y_val': y_val}.items():
            df.to_csv(config['data_dir'] / 'cache' / 'initial_data_split' / (file_name + '.csv'), index=False)

    [X_train, y_train, X_val, y_val, X_test, y_test] = list(
        map(lambda t: t.reset_index(drop=True), [X_train, y_train, X_val, y_val, X_test, y_test])
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_train_val_test(kind=None):
    dfs = []
    for file_name in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:

        if kind == 'classification_on_initial':
            df = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / (file_name + '.csv'))
            if 'X' in file_name:
                df = df[config['classification']['X_columns']]

        elif kind == 'classification_on_reconstructed':
            if 'X' in file_name:
                df = pd.read_csv(config['data_dir'] / 'cache' / 'models' / config['compression']['compressor_name'] / (
                        'reconstructed_' + file_name + '.csv'))
                df = df[config['classification']['X_columns']]
            else:
                df = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / (file_name + '.csv'))

        elif kind == 'compression':
            df = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / (file_name + '.csv'))

        elif kind == 'generation' or 'decompression':
            df = None
        else:
            raise NotImplementedError

        dfs.append(df)

    if kind == 'generation':
        dfs = []
        l = LabelBinarizer()
        df = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / 'y_train.csv')
        l.fit(df)

        for file_name in ['train', 'val', 'test']:
            df = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / ('X_' + file_name + '.csv'))
            df_ = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / ('y_' + file_name + '.csv'))
            dfs.append(pd.concat([pd.DataFrame(l.transform(df_)), df[config['generation']['conditional_columns']]], axis=1))

            df = pd.read_csv(config['data_dir'] / 'cache' / 'models' / config['compression']['compressor_name'] / ('compressed_X_' + file_name + '.csv'))[config['generation']['compressed_columns']]
            dfs.append(df)

    if kind == 'decompression':
        dfs = []
        l = LabelBinarizer()
        df = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / 'y_train.csv')
        l.fit(df)

        for file_name in ['train', 'val', 'test']:
            df = pd.read_csv(config['data_dir'] / 'cache' / 'models' / config['generation']['generator_name'] / (
                        'generated_compressed_' + file_name + '.csv'))[config['generation']['compressed_columns']]
            dfs.append(df)

            df = pd.read_csv(config['data_dir'] / 'cache' / 'initial_data_split' / ('y_' + file_name + '.csv'))
            dfs.append(df)



    return (dfs[0], dfs[1]), (dfs[2], dfs[3]), (dfs[4], dfs[5])
