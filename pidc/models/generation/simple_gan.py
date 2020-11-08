from .base import BaseGenerator
from config import config
import pytorch_lightning as pl
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import os


class SimpleGAN(BaseGenerator):
    def __init__(self, do_train=True):

        if do_train:
           self.model = SimpleGAN_()

        self.trainer = pl.Trainer(
            gpus=config['generation']['gpus'],
            max_epochs=config['generation']['max_epochs'],
        )

        self.model_name = config['generation']['generator_name']

    def fit(self, cond_train, data_train, **kwargs):
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(cond_train.values), torch.Tensor(data_train.values)
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['generation']['batch_size'],
            num_workers=config['generation']['num_workers'],
            shuffle=True
        )

        self.trainer.fit(self.model, train_dataloader)

    def generate(self, cond, save=True, filename_suffix=None, **kwargs):
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(cond.values), torch.Tensor(cond.index.tolist())
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['generation']['batch_size'],
            num_workers=config['generation']['num_workers']
        )

        self.model.test_predictions_buffer = pd.DataFrame()
        self.trainer.test(self.model, dataloader)

        preds = self.model.test_predictions_buffer.sort_index()

        if save:
            preds_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
            if not os.path.isdir(preds_dir_path):
                os.mkdir(preds_dir_path)
            if filename_suffix:
                filename = f'generated_compressed_{filename_suffix}.csv'
            else:
                filename = f'generated_compressed.csv'

            preds.to_csv(preds_dir_path / filename, index=False)

        return preds

    def decompress(self, compr, ae, save=True, filename_suffix=None, **kwargs):
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(compr.values), torch.Tensor(compr.index.tolist())
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['compression']['batch_size'],
            num_workers=config['compression']['num_workers']
        )

        ae.model.compress = False
        ae.model.test_predictions_compressed_buffer = pd.DataFrame()
        ae.model.test_predictions_reconstructed_buffer = pd.DataFrame()
        ae.trainer.test(ae.model, dataloader)

        preds = ae.model.test_predictions_reconstructed_buffer.sort_index()

        if save:
            preds_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
            if not os.path.isdir(preds_dir_path):
                os.mkdir(preds_dir_path)
            if filename_suffix:
                filename = f'generated_reconstructed_{filename_suffix}.csv'
            else:
                filename = f'generated_reconstructed.csv'

            preds.to_csv(preds_dir_path / filename, index=False)

        return preds

    def save(self):
        model_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
        if not os.path.isdir(model_dir_path):
            os.mkdir(model_dir_path)

        self.trainer.save_checkpoint(str(model_dir_path / 'model.ckpt'))

    @classmethod
    def load(cls):
        self = SimpleGAN(do_train=False)
        model_dir_path = config['data_dir'] / 'cache' / 'models' / config['generation']['generator_name']
        self.model = SimpleGAN_.load_from_checkpoint(str(model_dir_path / 'model.ckpt'))

        return self


class SimpleGenerator_(nn.Module):
    def __init__(self, cond_dim, latent_dim, data_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim+cond_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, data_dim),
            nn.Tanh()
        )

    def forward(self, z_and_cond):
        z_and_cond = torch.cat(z_and_cond, 1)
        data = self.model(z_and_cond)

        return data


class SimpleDiscriminator_(nn.Module):
    def __init__(self, cond_dim, data_dim):
        super().__init__()
        self.cond_dim = cond_dim
        self.data_dim = data_dim

        self.model = nn.Sequential(
            nn.Linear(data_dim+cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, data_and_cond):
        # print(len(data_and_cond), '    ', data_and_cond[0].shape, '   ', data_and_cond[1].shape)
        data_and_cond = torch.cat(data_and_cond, 1)
        validity = self.model(data_and_cond)

        return validity


class SimpleGAN_(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.cond_dim = len(config['generation']['conditional_columns']) + config['data_import']['target_column_classes']
        self.latent_dim = config['generation']['latent_dim']
        self.data_dim = config['compression']['latent_dim']

        self.generator = SimpleGenerator_(cond_dim=self.cond_dim, latent_dim=self.latent_dim, data_dim=self.data_dim)
        self.discriminator = SimpleDiscriminator_(cond_dim=self.cond_dim, data_dim=self.data_dim)

    def forward(self, z_and_cond):
        return self.generator(z_and_cond)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        cond, data = batch

        z = torch.randn(cond.shape[0], self.latent_dim)
        z = z.type_as(cond)

        if optimizer_idx == 0:

            valid = torch.ones(cond.size(0), 1)
            valid = valid.type_as(cond)

            g_loss = self.adversarial_loss(self.discriminator([self([z, cond]), cond]), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            valid = torch.ones(cond.size(0), 1)
            valid = valid.type_as(cond)
            real_loss = self.adversarial_loss(self.discriminator([data, cond]), valid)
            fake = torch.zeros(data.size(0), 1)
            fake = fake.type_as(data)
            fake_loss = self.adversarial_loss(
                self.discriminator([self([z, cond]).detach(), cond]), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def test_step(self, batch, batch_idx):
        cond, ids = batch
        z = torch.randn(cond.shape[0], self.latent_dim)
        z = z.type_as(cond)
        preds = self.generator([z, cond])

        prediction_to_append = pd.DataFrame(data=preds.cpu().numpy(),
                                                       index=ids.cpu().numpy().astype(int),
                                                       columns=config['generation']['compressed_columns'])
        self.test_predictions_buffer = self.test_predictions_buffer.append(prediction_to_append)

    def configure_optimizers(self):

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=config['generation']['lr'])
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=config['generation']['lr'])
        return [opt_g, opt_d], []

