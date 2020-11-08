from config import config
from pidc.models.generation.simple_gan import SimpleGAN
from pidc.utils.data import load_train_val_test


def main():
    (cond_train, data_train), (cond_val, data_val), (cond_test, data_test) = load_train_val_test(kind='generation')

    model = SimpleGAN()
    model.fit(cond_train, data_train)
    model.save()


if __name__ == '__main__':
    main()
