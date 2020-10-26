from .base import BaseCompressor
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import reduce
import pandas as pd
import os
from config import config


class AE(BaseCompressor):
    def __init__(self, model_name='ae'):
        self.model = AE_(50, 3)
        self.trainer = pl.Trainer(gpus=[1], max_epochs=10000)

        self.model_name = model_name

    def fit(self, X_train, y_train, X_val, y_val, **kwargs):
        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train.values), torch.Tensor(X_train.index.tolist())
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(1e5),
            shuffle=True,
            num_workers=24
        )

        val_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_val.values), torch.Tensor(X_val.index.tolist())
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=int(1e5),
            num_workers=24
        )

        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, X, save=True, filename_suffix=None, **kwargs):
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X.values), torch.Tensor(X.index.tolist())
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(1e5),
            num_workers=24
        )

        self.trainer.test(self.model, dataloader)
        preds = self.model.test_predictions_buffer.sort_index()

        if save:
            preds_dir_path = config['data_dir'] / 'output' / 'models' / self.model_name
            if not os.path.isdir(preds_dir_path):
                os.mkdir(preds_dir_path)
            if filename_suffix:
                filename = f'predictions_{filename_suffix}.csv'
            else:
                filename = f'predictions.csv'
            preds.to_csv(preds_dir_path / filename, index=False)

        return preds

    def save(self):
        model_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
        if not os.path.isdir(model_dir_path):
            os.mkdir(model_dir_path)

        self.trainer.save_checkpoint(str(model_dir_path / 'model.ckpt'))

    @classmethod
    def load(cls, model_name):
        self = AE(model_name=model_name)
        model_dir_path = config['data_dir'] / 'cache' / 'models' / self.model_name
        self.model = AE_.load_from_checkpoint(str(model_dir_path / 'model.ckpt'))

        return self


class AE_(pl.LightningModule):

    def __init__(
            self,
            input_dim,
            latent_dim,
            n_hidden_layers=4,
            hidden_layer_size=512,
            lr=1e-4,
            **kwargs
    ):
        super(AE_, self).__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr

        self.test_predictions_buffer = pd.DataFrame()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.LeakyReLU(),
            *reduce(
                lambda x, y: x + y,
                [[nn.Linear(hidden_layer_size, hidden_layer_size), nn.LeakyReLU()] for _ in range(n_hidden_layers)])
            ,
            nn.Linear(hidden_layer_size, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_layer_size),
            nn.LeakyReLU(),
            *reduce(
                lambda x, y: x + y,
                [[nn.Linear(hidden_layer_size, hidden_layer_size), nn.LeakyReLU()] for _ in range(n_hidden_layers)]
            )
            ,
            nn.Linear(hidden_layer_size, input_dim),
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
        preds = self.forward(x)
        prediction_to_append = pd.DataFrame(data=preds.cpu().numpy(),
                                            index=ids.cpu().numpy().astype(int),
                                            columns=[f'encoded_feature_{i + 1}' for i in range(self.latent_dim)])
        self.test_predictions_buffer = self.test_predictions_buffer.append(prediction_to_append)

        return {}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack(
            [output['val_loss'] for output in outputs]
        ).mean()

        print(avg_val_loss.item())
        return {'avg_val_loss': avg_val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)