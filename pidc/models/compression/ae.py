from config import config
from .base import BaseCompressor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer, seed_everything
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import reduce
import pandas as pd
import os


class AE(BaseCompressor):
    def __init__(self, do_train=True):

        seed_everything(42)
        if do_train:
            self.model = AE_(len(config['data_import']['X_columns']),
                             config['compression']['latent_dim'],
                             config['compression']['n_hidden_layers'],
                             config['compression']['hidden_layer_size'],
                             config['compression']['dropout_p'],
                             config['compression']['lr'])

        early_stopping_callback = EarlyStopping(
            monitor='avg_val_loss',
            min_delta=config['compression']['early_stopping_min_delta'],
            patience=config['compression']['early_stopping_patience'],
            verbose=True,
            mode='min'
        )
        self.trainer = pl.Trainer(
            gpus=config['compression']['gpus'],
            max_epochs=config['compression']['max_epochs'],
            early_stop_callback=early_stopping_callback
        )

        self.model_name = config['compression']['compressor_name']

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train.values), torch.Tensor(X_train.index.tolist())
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['compression']['batch_size'],
            num_workers=config['compression']['num_workers'],
            shuffle=True
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_val.values), torch.Tensor(X_val.index.tolist())
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['compression']['batch_size'],
            num_workers=config['compression']['num_workers']
        )

        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, X, save=True, filename_suffix=None, **kwargs):
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X.values), torch.Tensor(X.index.tolist())
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config['compression']['batch_size'],
            num_workers=config['compression']['num_workers']
        )

        self.model.test_predictions_compressed_buffer = pd.DataFrame()
        self.model.test_predictions_reconstructed_buffer = pd.DataFrame()
        self.trainer.test(self.model, dataloader)

        preds_compressed = self.model.test_predictions_compressed_buffer.sort_index()
        preds_reconstructed = self.model.test_predictions_reconstructed_buffer.sort_index()

        if save:
            preds_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
            if not os.path.isdir(preds_dir_path):
                os.mkdir(preds_dir_path)
            if filename_suffix:
                filename_compressed = f'compressed_{filename_suffix}.csv'
                filename_reconstructed = f'reconstructed_{filename_suffix}.csv'
            else:
                filename_compressed = f'compressed.csv'
                filename_reconstructed = f'reconstructed.csv'

            preds_compressed.to_csv(preds_dir_path / filename_compressed, index=False)
            preds_reconstructed.to_csv(preds_dir_path / filename_reconstructed, index=False)

        return preds_compressed, preds_reconstructed

    def save(self):
        model_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
        if not os.path.isdir(model_dir_path):
            os.mkdir(model_dir_path)

        self.trainer.save_checkpoint(str(model_dir_path / 'model.ckpt'))

    @classmethod
    def load(cls):
        self = AE(do_train=False)
        model_dir_path = config['data_dir'] / 'cache' / 'models' / config['compression']['compressor_name']
        self.model = AE_.load_from_checkpoint(str(model_dir_path / 'model.ckpt'))

        return self


class AE_(pl.LightningModule):

    def __init__(
            self,
            input_dim,
            latent_dim,
            n_hidden_layers,
            hidden_layer_size,
            dropout_p,
            lr,
    ):
        super(AE_, self).__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr

        self.test_predictions_compressed_buffer = pd.DataFrame()
        self.test_predictions_reconstructed_buffer = pd.DataFrame()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            *reduce(
                lambda x, y: x + y,
                [[nn.Linear(hidden_layer_size, hidden_layer_size), nn.LeakyReLU(), nn.BatchNorm1d(hidden_layer_size),
                  nn.Dropout(dropout_p)] for _ in range(n_hidden_layers)]),
            nn.Linear(hidden_layer_size, latent_dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_layer_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            *reduce(
                lambda x, y: x + y,
                [[nn.Linear(hidden_layer_size, hidden_layer_size), nn.LeakyReLU(), nn.BatchNorm1d(hidden_layer_size),
                  nn.Dropout(dropout_p)] for _ in range(n_hidden_layers)]),
            nn.Linear(hidden_layer_size, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)

        return z

    def step(self, batch, batch_idx):
        x, ids = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, ids = batch
        preds_compressed = self.encoder(x)
        preds_reconstructed = self.decoder(preds_compressed)

        prediction_compressed_to_append = pd.DataFrame(data=preds_compressed.cpu().numpy(),
                                                       index=ids.cpu().numpy().astype(int),
                                                       columns=[f'encoded_feature_{i + 1}' for i in
                                                                range(self.latent_dim)])
        self.test_predictions_compressed_buffer = self.test_predictions_compressed_buffer.append(
            prediction_compressed_to_append)

        prediction_reconstructed_to_append = pd.DataFrame(data=preds_reconstructed.cpu().numpy(),
                                                          index=ids.cpu().numpy().astype(int),
                                                          columns=config['data_import']['X_columns'])
        self.test_predictions_reconstructed_buffer = self.test_predictions_reconstructed_buffer.append(
            prediction_reconstructed_to_append)

        return {}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack(
            [output['val_loss'] for output in outputs]
        ).mean()

        return {'avg_val_loss': avg_val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
