import numpy as np
import torch
from torch.optim import Adam, RMSprop, SGD, AdamW
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle
import argparse
import random
import wandb
import os
import constants
from dataset import load_data
from models.trainer import ModelTrainer
import models.load_model as load_model

os.environ["WANDB_API_KEY"] = "d5a82a7201d64dd1120fa3be37072e9e06e382a1"
os.environ['WANDB_START_METHOD'] = 'thread'
cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
local_data_path = "/Users/nonarajabi/Desktop/KTH/Smell/Novel_Bulb_measure/data/"
local_save_path = "/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/"


def seed_everything(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data', type=str, default='dataset3_tfr')
    parser.add_argument('--tmin', type=float, default=None)
    parser.add_argument('--tmax', type=float, default=None)
    parser.add_argument('--ebg_transform', type=str, default='tfr_morlet')
    parser.add_argument('--subject_id', type=int, default=0)
    parser.add_argument('--eeg', type=str, default='lstm')
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--num_layers', type=str, default=None)
    parser.add_argument('--optim_name', type=str, default='adamw')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with wandb.init(project="EBG_Olfaction", config=args):
    # with wandb.init():
        args = wandb.config
        seed = args.seed
        seed_everything(seed)
        dataset_name = args.data
        subject_id = args.subject_id
        eeg_enc_name = args.eeg
        batch_size = args.batch_size
        lr = args.lr
        epochs = args.epoch
        optim_name = args.optim_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device = ", device)

        if dataset_name == 'dataset3_tfr':
            local_data_path = '/Users/nonarajabi/Desktop/KTH/Smell/paper3/TFRs/'

        paths = {
            "eeg_data": cluster_data_path if device == 'cuda' else local_data_path,
            "save_path": cluster_save_path if device == 'cuda' else local_save_path
        }

        constants.data_constants['tmin'] = args.tmin
        constants.data_constants['tmax'] = args.tmax
        constants.data_constants['ebg_transform'] = args.ebg_transform

        data, weights, n_time_samples = load_data.load(dataset_name=dataset_name,
                                                       path=paths['eeg_data'],
                                                       batch_size=batch_size,
                                                       seed=seed,
                                                       device=device, **constants.data_constants)

        # constants.model_constants['lstm']['input_size'] = data.f_max - data.f_min

        if args.dropout is not None:
            constants.model_constants['lstm']['dropout'] = args.dropout
            constants.model_constants['rnn']['dropout'] = args.dropout
        if args.hidden_size is not None:
            constants.model_constants['lstm']['hidden_size'] = args.hidden_size
            constants.model_constants['rnn']['hidden_size'] = args.hidden_size
        if args.num_layers is not None:
            constants.model_constants['lstm']['num_layers'] = args.num_layers
            constants.model_constants['rnn']['num_layers'] = args.num_layers

        constants.model_constants['eegnet']['n_samples'] = n_time_samples
        constants.model_constants['eegnet_attention']['n_samples'] = n_time_samples
        
        # sss = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
        # train_loader, val_loader, test_loader = loaders['train_loader'], loaders['val_loader'], loaders['test_loader']

        metrics = {'loss': [], 'acc': [], 'auroc': []}
        g = torch.Generator().manual_seed(seed)
        # for fold, (train_ids, test_ids) in enumerate(sss.split(data, data.labels)):

        #     print(f'--------------- Fold {fold} ---------------')

        #     # Sample elements randomly from a given list of ids, no replacement.
        #     train_sub_sampler = torch.utils.data.SubsetRandomSampler(train_ids, generator=g)
        #     test_sub_sampler = torch.utils.data.SubsetRandomSampler(test_ids, generator=g)

            # Define data loaders for training and testing data in this fold
        train_size = int(0.8*len(data))
        val_size = len(data) - train_size
        train_data, val_data = torch.utils.data.random_split(
        data, [train_size, val_size], generator=g)
    
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size, generator=g)
        val_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size, generator=g)

        model = load_model.load(eeg_enc_name, **constants.model_constants[eeg_enc_name])
        print(f"Training {eeg_enc_name} on {dataset_name} with {constants.model_constants[eeg_enc_name]}")
        model = model.double()
        if optim_name == 'adam':
            optim = Adam(model.parameters(), lr=lr, weight_decay=0.05)
        elif optim_name == 'adamw':
            optim = AdamW(model.parameters(), lr=lr, weight_decay=0.1, amsgrad=True)
        elif optim_name == 'rmsprop':
            optim = RMSprop(model.parameters(), lr=lr, weight_decay=0.05)
        elif optim_name == 'sgd':
            optim = SGD(model.parameters(), lr=lr, momentum=0.1, weight_decay=0.05)
        else:
            raise NotImplementedError
        trainer = ModelTrainer(model=model, optimizer=optim, n_epochs=epochs, save_path=paths['save_path'],
                               weights=weights, device=device)
        best_model = trainer.train(train_loader, val_loader)
        metrics['loss'].append(best_model['loss'])
        metrics['acc'].append(best_model['acc'])
        metrics['auroc'].append(best_model['auroc'])
            # model.load_state_dict(best_model['model_state_dict'])
        print(f"Average Balanced Accuracy: {np.mean(np.array(metrics['acc']))}")
        with open(os.path.join(paths['save_path'], f'{eeg_enc_name}_{batch_size}_{lr}_{epochs}_{optim_name}.pkl'),
                  'wb') as f:
            pickle.dump(metrics, f)
        # loss_test, acc_test, auroc_test = trainer.evaluate(model, test_loader)
