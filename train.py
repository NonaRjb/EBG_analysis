import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam, RMSprop, SGD, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold
import torchvision.transforms as transforms
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
from dataset.data_utils import RandomNoise, RandomMask, TemporalJitter

os.environ["WANDB_API_KEY"] = "d5a82a7201d64dd1120fa3be37072e9e06e382a1"
os.environ['WANDB_START_METHOD'] = 'thread'
cluster_data_path = '/local_storage/datasets/nonar/ebg/'
cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
# cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
# cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/"

data_transforms = transforms.Compose([
    # MinMaxNormalize(),
    # transforms.ToTensor(),
    RandomNoise(mean=None, std=None, p=0.6),  # p = 0.6 (for 'both' and 'ebg')
    RandomMask(ratio=0.75, p=0.4),  # ratio=0.75 and p=0.3 for 'both', ratio=0.6 and p=0.3 for 'ebg'
    TemporalJitter(max_jitter=100, p=0.4)
])


class WarmUpLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, start_lr, target_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.delta_lr = (target_lr - start_lr) / warmup_steps
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.start_lr + self.delta_lr * self.last_epoch for _ in self.optimizer.param_groups]
        else:
            return [base_lr for base_lr in self.base_lrs]


def train_subject(subject_data):
    # wandb.init()

    subject, eeg_enc_name, dataset_name, epochs, seed, split_seed, directory_name, device, constants = subject_data

    print(f"********** subject {subject} **********")

    weight_decay = constants.training_constants['weight_decay']
    lr = constants.training_constants['lr']
    scheduler_name = constants.training_constants['scheduler_name']
    optim_name = constants.training_constants['optim_name']
    batch_size = constants.training_constants['batch_size']
    # fold = str(constants.training_constants['fold'])+".pkl"
    # i = constants.training_constants['fold'] - 1

    paths = {
        "eeg_data": cluster_data_path if device == 'cuda' else local_data_path,
        "save_path": cluster_save_path if device == 'cuda' else local_save_path
    }
    splits_path = os.path.join(paths['eeg_data'], "splits_ebg4")
    os.makedirs(os.path.join(paths['save_path'], directory_name, str(subject)), exist_ok=True)
    # os.makedirs(os.path.join(paths['save_path'], directory_name, str(subject_id), str(args.tmin)),
    #             exist_ok=True)
    paths['save_path'] = os.path.join(paths['save_path'], directory_name, str(subject))  # , str(args.tmin))
    print(f"Directory '{directory_name}' created.")
    # create_readme(wandb.config, path=paths['save_path']
    # data, weights, n_time_samples = load_data.load(
    _, data, n_time_samples = load_data.load(
        dataset_name=dataset_name,
        path=paths['eeg_data'],
        batch_size=batch_size,
        seed=seed,
        split_seed=split_seed,
        augmentation=False,
        subject_id=subject,
        transform=None,
        device=device, **constants.data_constants
    )

    _, transformed_data, _ = load_data.load(
        dataset_name=dataset_name,
        path=paths['eeg_data'],
        batch_size=batch_size,
        seed=seed,
        split_seed=split_seed,
        augmentation=False,
        subject_id=subject,
        transform=data_transforms,
        device=device, **constants.data_constants
    )

    # constants.model_constants['lstm']['input_size'] = data.f_max - data.f_min
    print("EEG sequence length = ", n_time_samples)
    print(f"Training on {constants.data_constants['n_classes']} classes")

    data.labels = np.array(data.labels)

    g = torch.Generator().manual_seed(seed)
    skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=seed)
    metrics = {'loss': [], 'acc': [], 'auroc': [], 'epoch': [], 'val_auroc': []}
    for i, fold in enumerate(os.listdir(os.path.join(splits_path, str(subject)))):
    # for i, (train_index, val_index) in enumerate(skf.split(data.data, data.labels)):
        with open(os.path.join(splits_path, str(subject), fold), 'rb') as f:
            split = pickle.load(f)
        train_index = split['train']
        val_index = split['val']
        test_index = split['test']
        val_labels = data.labels[val_index]
        # test_labels = data.labels[test_index]
        # if np.sum(val_labels == 1) == 0:  # or np.sum(test_labels == 1) == 0:
        #     continue
        # Sample elements randomly from a given list of ids, no replacement.
        train_sub_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=g)
        val_sub_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=g)
        test_sub_sampler = torch.utils.data.SubsetRandomSampler(test_index, generator=g)
        # Create DataLoader for training and validation
        train_loader = DataLoader(transformed_data, batch_size=batch_size, sampler=train_sub_sampler, drop_last=True)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sub_sampler, drop_last=False)
        test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sub_sampler, drop_last=False)
        model = load_model.load(eeg_enc_name, **constants.model_constants[eeg_enc_name])
        # print(model)
        # print(f"Training {eeg_enc_name} on {dataset_name} with {constants.model_constants[eeg_enc_name]}")
        model = model.double()
        optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        if scheduler_name == 'plateau':
            # warmup_scheduler = WarmUpLR(
            #     optim, warmup_steps=constants.training_constants['warmup_steps'], 
            #     start_lr=0.000001, target_lr=0.00005)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                                   patience=constants.training_constants['patience'],
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
                               weights=None, device=device, scheduler=scheduler, warmup=None,
                               warmup_steps=constants.training_constants['warmup_steps'], batch_size=batch_size)
        best_model = trainer.train(train_loader, val_loader)
        model.load_state_dict(best_model['model_state_dict'])
        model.eval()
        test_loss, test_acc, test_auroc, y_true_test, y_pred_test = trainer.evaluate(model, test_loader)
        # test_loss, test_acc, test_auroc, y_true_test, y_pred_test = trainer.evaluate(model, val_loader)
        print(
            f"Best Val Loss = {best_model['loss']}, AUC Score = {best_model['auroc']} (Epoch = {best_model['epoch']})")
        print(f"Test AUC Score = {test_auroc}")
        # metrics['loss'].append(test_loss)
        # metrics['acc'].append(test_acc)
        metrics['auroc'].append(test_auroc)
        # metrics['epoch'].append(best_model['epoch'])
        # metrics['val_auroc'].append(best_model['auroc'])
        metrics['loss'].append(best_model['loss'])
        metrics['acc'].append(best_model['acc'])
        metrics['epoch'].append(best_model['epoch'])
        # metrics['auroc'].append(best_model['auroc'])
        with open(os.path.join(paths['save_path'], f'{i + 1}.pkl'),
                  'wb') as f:
            pickle.dump({'auroc': test_auroc, 'epoch': best_model['epoch']}, f)
        del model
    print("Median AUC = ", np.median(np.asarray(metrics['auroc'])))


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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=1000)
    # parser.add_argument('--fold', type=int, default=1)
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

    constants.training_constants['lr'] = args.lr
    constants.training_constants['patience'] = args.patience
    constants.data_constants['tmin'] = args.tmin
    constants.data_constants['tmax'] = args.tmax
    constants.data_constants['w'] = args.w
    constants.data_constants['ebg_transform'] = args.ebg_transform

    if constants.data_constants['modality'] == "source":
        directory_name = f"{dataset_name}_{eeg_enc_name}_lr{args.lr}_pat{args.patience}"
    else:
        directory_name = \
            f"{dataset_name}_{eeg_enc_name}_{constants.data_constants['modality']}_lr{args.lr}_pat{args.patience}"
    os.makedirs(os.path.join(main_paths['save_path'], directory_name), exist_ok=True)

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
        transform=None,
        device=device, **constants.data_constants
    )

    # constants.training_constants['fold'] = args.fold
    constants.model_constants['eegnet']['n_samples'] = n_time_samples
    constants.model_constants['eegnet1d']['n_samples'] = n_time_samples
    constants.model_constants['eegnet_attention']['n_samples'] = n_time_samples
    constants.model_constants['resnet1d']['n_samples'] = n_time_samples
    constants.model_constants['resnet1d']['net_seq_length'][0] = n_time_samples
    subject_data_list = [(sid, eeg_enc_name, dataset_name, epochs, seed, split_seed, directory_name, device, constants)
                         for sid in subject_ids]
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
