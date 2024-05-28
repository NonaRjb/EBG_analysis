import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam, RMSprop, SGD, AdamW
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle
import argparse
import random
import wandb
import os
from datetime import datetime
# import constants
from config import DNNConfig
from dataset import load_data
from models.trainer import ModelTrainer
import models.load_model as load_model

os.environ["WANDB_API_KEY"] = "d5a82a7201d64dd1120fa3be37072e9e06e382a1"
os.environ['WANDB_START_METHOD'] = 'thread'
cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
# cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
# cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/"


def dummy_function(a):
    x1 = torch.randn((10000, 10000)).to("cuda:0")
    while True:
        _ = x1 * x1
    

def train_subject(subject_data):

    # wandb.init()
   
    subject, eeg_enc_name, dataset_name, epochs, seed, split_seed, directory_name, device, constants = subject_data

    print(f"********** subject {subject} **********")

    weight_decay = constants.training_constants['weight_decay']
    lr = constants.training_constants['lr']
    scheduler_name = constants.training_constants['scheduler_name']
    optim_name = constants.training_constants['optim_name']
    batch_size = constants.training_constants['batch_size']
    
    paths = {
                "eeg_data": cluster_data_path if device == 'cuda' else local_data_path,
                "save_path": cluster_save_path if device == 'cuda' else local_save_path
            }
    os.makedirs(os.path.join(paths['save_path'], directory_name, str(subject)), exist_ok=True)
    # os.makedirs(os.path.join(paths['save_path'], directory_name, str(subject_id), str(args.tmin)),
    #             exist_ok=True)
    paths['save_path'] = os.path.join(paths['save_path'], directory_name, str(subject)) # , str(args.tmin))
    print(f"Directory '{directory_name}' created.")
    # create_readme(wandb.config, path=paths['save_path']
    # data, weights, n_time_samples = load_data.load(
    loaders, data, n_time_samples = load_data.load(
        dataset_name=dataset_name,
        path=paths['eeg_data'],
        batch_size=batch_size,
        seed=seed,
        split_seed=split_seed,
        augmentation=False,
        subject_id=subject,
        device=device, **constants.data_constants
    )
    # constants.model_constants['lstm']['input_size'] = data.f_max - data.f_min
    print("EEG sequence length = ", n_time_samples)
    print(f"Training on {constants.data_constants['n_classes']} classes")

    train_loader, val_loader, test_loader = loaders['train_loader'], loaders['val_loader'], loaders['test_loader']

    metrics = {'loss': [], 'acc': [], 'auroc': [], 'epoch': []}
    model = load_model.load(eeg_enc_name, **constants.model_constants[eeg_enc_name])
    print(model)
    print(f"Training {eeg_enc_name} on {dataset_name} with {constants.model_constants[eeg_enc_name]}")
    model = model.double()
    if optim_name == 'adam':
        optim = Adam(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_name == 'adamw':
        optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim_name == 'rmsprop':
        optim = RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'sgd':
        optim = SGD(model.parameters(), lr=lr, momentum=0.1, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    if scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=200,
                                                               min_lr=0.1 * 1e-7,
                                                               factor=0.1)
    elif scheduler_name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[16, 64, 256],
                                                         gamma=0.1)
    elif scheduler_name == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99, last_epoch=-1)
    elif scheduler_name == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1., end_factor=0.5, total_iters=30)
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0, last_epoch=-1)
    else:
        raise NotImplementedError
    trainer = ModelTrainer(model=model, optimizer=optim, n_epochs=epochs,
                           n_classes=constants.data_constants['n_classes'], save_path=paths['save_path'],
                           weights=None, device=device, scheduler=scheduler)
    best_model = trainer.train(train_loader, val_loader)
    metrics['loss'].append(best_model['loss'])
    metrics['acc'].append(best_model['acc'])
    metrics['auroc'].append(best_model['auroc'])
    metrics['epoch'].append(best_model['epoch'])
    # model.load_state_dict(best_model['model_state_dict'])
    print(f"Best Validation AUC Score = {best_model['auroc']} (Epoch = {best_model['epoch']})")
    with open(os.path.join(paths['save_path'], f'{split_seed}.pkl'),
              'wb') as f:
        pickle.dump(metrics, f)


def seed_everything(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)


def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data', type=str, default='ebg4_source')
    parser.add_argument('--tmin', type=float, default=None)
    parser.add_argument('--tmax', type=float, default=None)
    parser.add_argument('-w', type=float, default=None)
    parser.add_argument('--ebg_transform', type=str, default='tfr_morlet')
    parser.add_argument('--subject_id', type=int, default=0)
    parser.add_argument('--eeg', type=str, default='eegnet1d')
    # parser.add_argument('--optim_name', type=str, default='adamw')
    # parser.add_argument('-b', '--batch_size', type=int, default=32)
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--lr_scheduler', type=str, default='plateau')
    # parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project="EBG_Olfaction", config=args)
        # with wandb.init():
    # args = wandb.config
    constants = DNNConfig()
    eeg_enc_name = args.eeg
    epochs = args.epoch
    seed = args.seed
    split_seed = args.split_seed
    seed_everything(seed)
    dataset_name = args.data
    if args.subject_id != -1:
        subject_ids = [args.subject_id]
    else:
        if 'ebg4' in dataset_name:
            if 'source' in dataset_name or 'source' in constants.data_constants['modality']:
                subject_ids = [i for i in range(1, 54) if i not in [3, 10]]
            else:
                subject_ids = [i for i in range(1, 54) if i != 10]
        elif 'ebg1' in dataset_name:
            subject_ids = [i for i in range(1, 31) if i != 4]
        else:
            raise NotImplementedError
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = ", device)
    # if dataset_name == 'ebg3_tfr':
    #     local_data_path = '/Users/nonarajabi/Desktop/KTH/Smell/paper3/TFRs/'
    main_paths = {
        "eeg_data": cluster_data_path if device == 'cuda' else local_data_path,
        "save_path": cluster_save_path if device == 'cuda' else local_save_path
    }
    if constants.data_constants['modality'] == "source":
        directory_name = f"{dataset_name}_{eeg_enc_name}"
    else:
        directory_name = f"{dataset_name}_{eeg_enc_name}_{constants.data_constants['modality']}"
    os.makedirs(os.path.join(main_paths['save_path'], directory_name), exist_ok=True)
    constants.data_constants['tmin'] = args.tmin
    constants.data_constants['tmax'] = args.tmax
    constants.data_constants['w'] = args.w
    constants.data_constants['ebg_transform'] = args.ebg_transform
    if constants.data_constants['modality'] == "source":
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 4
    elif constants.data_constants['modality'] == "ebg":
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 4
    elif constants.data_constants['modality'] == "eeg":
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 63
    elif constants.data_constants['modality'] in ["ebg-sniff", "sniff-ebg"]:
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 5
    elif constants.data_constants['modality'] in ["eeg-sniff", "sniff-eeg"]:
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 64
    elif constants.data_constants['modality'] == "both-sniff":
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 68
    elif constants.data_constants['modality'] == 'sniff':
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 1
    elif constants.data_constants['modality'] in ['source-sniff', 'sniff-source']:
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 5
    elif constants.data_constants['modality'] in ['source-ebg', 'ebg-source']:
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 8
    elif constants.data_constants['modality'] in ['source-eeg', 'eeg-source']:
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 67
    elif constants.data_constants['modality'] in ['eeg-ebg', 'ebg-eeg']:
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 67
    else:
        raise NotImplementedError
    # load a sample subject's data to compute the number of time samples
    _, _, n_time_samples = load_data.load(
        dataset_name=dataset_name,
        path=main_paths['eeg_data'],
        batch_size=constants.training_constants['batch_size'],
        seed=seed,
        split_seed=split_seed,
        augmentation=False,
        subject_id=1,
        device=device, **constants.data_constants
    )
    
    constants.model_constants['eegnet']['n_samples'] = n_time_samples
    constants.model_constants['eegnet1d']['n_samples'] = n_time_samples
    constants.model_constants['eegnet_attention']['n_samples'] = n_time_samples
    constants.model_constants['resnet1d']['n_samples'] = n_time_samples
    subject_data_list = [(sid, eeg_enc_name, dataset_name, epochs, seed, split_seed, directory_name, device, constants) for sid in subject_ids] 
    print("number of available CPUs = ", os.cpu_count())
    # num_processes = max(os.cpu_count(), 2)
    # num_processes = 4
    # with Pool(processes=num_processes) as pool:
    #     # pool.map_async(dummy_function, [1])
    #     pool.map(train_subject, subject_data_list)
    for x in subject_data_list:
        train_subject(x)

    wandb.finish()

            
if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    main()