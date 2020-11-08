from config import config
from pidc.models.generation.simple_gan import SimpleGAN
from pidc.models.compression.ae import AE
from pidc.utils.data import load_train_val_test


def main():
    # (cond_train, data_train), (cond_val, data_val), (cond_test, data_test) = load_train_val_test(kind='generation')
    #
    model = SimpleGAN.load()
    # for name, cond in {'train': cond_train, 'val': cond_val, 'test': cond_test}.items():
    #     model.generate(cond, save=True, filename_suffix=f'{name}')
    #
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test(kind='decompression')
    model_ae = AE.load()
    for name, X in {'train': X_train, 'val': X_val, 'test': X_test}.items():
        model.decompress(X, model_ae, save=True, filename_suffix=f'{name}')

if __name__ == '__main__':
    main()
