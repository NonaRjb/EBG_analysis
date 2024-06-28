import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Adam, RMSprop, SGD, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import copy
import pickle
import argparse
import random
import wandb
import os
from tqdm import tqdm
from datetime import datetime
# import constants
from config import DNNConfig
# from dataset import load_data
from models.trainer import ModelTrainer, MultiModalTrainer
import models.load_model as load_model
from dataset.data_utils import RandomNoise, RandomMask, TemporalJitter, load_ebg4
from dataset.ebg4 import EBG4


os.environ["WANDB_API_KEY"] = "d5a82a7201d64dd1120fa3be37072e9e06e382a1"
os.environ['WANDB_START_METHOD'] = 'thread'
# cluster_data_path = '/local_storage/datasets/nonar/ebg/'
# cluster_save_path = '/Midgard/home/nonar/data/ebg/ebg_out/'
cluster_data_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/'
cluster_save_path = '/proj/berzelius-2023-338/users/x_nonra/data/Smell/plots/'
local_data_path = "/Volumes/T5 EVO/Smell/"
local_save_path = "/Users/nonarajabi/Desktop/KTH/Smell/ebg_out/"

time_windows = [(0.00, 0.25), (0.15, 0.40), (0.30, 0.55), (0.45, 0.70), (0.60, 0.85), (0.75, 1.0)]
data_transforms = transforms.Compose([
    # MinMaxNormalize(),
    # transforms.ToTensor(),
    RandomNoise(mean=None, std=None, p=0.6),  # p = 0.6 (for 'both' and 'ebg')
    RandomMask(ratio=0.75, p=0.4),  # ratio=0.75 and p=0.3 for 'both', ratio=0.6 and p=0.3 for 'ebg'
    TemporalJitter(max_jitter=100, p=0.4)
])
channel_dict = {
    'ebg': 4,
    'eeg': 64,
    'source': 4,
    'sniff': 1
}

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
    global time_windows
    subject, eeg_enc_name, dataset_name, data_type, epochs, seed, task, directory_name, device, constants = subject_data

    print(f"********** subject {subject} **********")

    weight_decay = constants.training_constants['weight_decay']
    lr = constants.training_constants['lr']
    scheduler_name = constants.training_constants['scheduler_name']
    optim_name = constants.training_constants['optim_name']
    batch_size = constants.training_constants['batch_size']
    # fold = str(constants.training_constants['fold'])+".pkl"
    # i = constants.training_constants['fold'] - 1

    if "whole_win" in task:
        time_windows = [(constants.data_constants['tmin'], constants.data_constants['tmax'])]

    paths = {
        "eeg_data": cluster_data_path if device == 'cuda' else local_data_path,
        "save_path": cluster_save_path if device == 'cuda' else local_save_path
    }
    splits_path = os.path.join(paths['eeg_data'], "splits_ebg4_with_test")
    os.makedirs(os.path.join(paths['save_path'], task, directory_name, str(subject)), exist_ok=True)
    # os.makedirs(os.path.join(paths['save_path'], directory_name, str(subject_id), str(args.tmin)),
    #             exist_ok=True)
    paths['save_path'] = os.path.join(paths['save_path'], task, directory_name, str(subject))  # , str(args.tmin))
    print(f"Directory '{directory_name}' created.")
    # create_readme(wandb.config, path=paths['save_path']
    # data, weights, n_time_samples = load_data.load(

    g = torch.Generator().manual_seed(seed)
    # outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    inner_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)

    data_array, labels_array, t, sfreq = load_ebg4(
        root=os.path.join(paths['eeg_data'], dataset_name),
        subject_id=subject,
        data_type=data_type,
        **constants.data_constants
    )

    data_dummy = EBG4(
                root_path=os.path.join(paths['eeg_data'], "ebg4"),
                source_data=data_array, label=labels_array, time_vec=t, fs=sfreq, 
                tmin=None, tmax=None, w=None, binary=constants.data_constants['binary'],
                data_type=data_type, modality=constants.data_constants["modality"], intensity=constants.data_constants['intensity'],
                pick_subjects=subject, fs_new=constants.data_constants['fs_new'], 
                normalize=constants.data_constants['normalize'], transform=None
                )
    data_dummy.labels = np.array(data_dummy.labels)

    metrics = {'loss': [], 'acc': [], 'auroc': [], 'epoch': [], 'test_auroc': []}
    # for i, fold in enumerate(os.listdir(os.path.join(splits_path, str(subject)))):
    for i, (train_all_index, test_index) in enumerate(outer_cv.split(data_dummy.data, data_dummy.labels)):
        best_models = []
        best_windows = []
        median_windows = []
        for win in time_windows:

            constants.data_constants['tmin'] = win[0]
            constants.data_constants['tmax'] = win[1]
            
            data = EBG4(
                root_path=os.path.join(paths['eeg_data'], "ebg4"),
                source_data=data_array, label=labels_array, time_vec=t, fs=sfreq, 
                tmin=win[0], tmax=win[1], w=None, binary=constants.data_constants['binary'],
                data_type=data_type, modality=constants.data_constants["modality"], intensity=constants.data_constants['intensity'],
                pick_subjects=subject, fs_new=constants.data_constants['fs_new'], 
                normalize=constants.data_constants['normalize'], transform=None
                )
            
            transformed_data = EBG4(
                root_path=os.path.join(paths['eeg_data'], "ebg4"),
                source_data=data_array, label=labels_array, time_vec=t, fs=sfreq, 
                tmin=win[0], tmax=win[1], w=None, binary=constants.data_constants['binary'],
                data_type=data_type, modality=constants.data_constants["modality"], intensity=constants.data_constants['intensity'],
                pick_subjects=subject, fs_new=constants.data_constants['fs_new'], 
                normalize=constants.data_constants['normalize'], transform=data_transforms
                )

            data.labels = np.array(data.labels)
            transformed_data.labels = np.array(transformed_data.labels)

            best_models_win = []
            for j, (train_index, val_index) in enumerate(inner_cv.split(data_dummy.data[train_all_index], data_dummy.labels[train_all_index])):
            
                train_index = train_all_index[train_index]
                val_index = train_all_index[val_index]
                    
                train_sub_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=g)
                val_sub_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=g)
                # Create DataLoader for training and validation
                train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sub_sampler, drop_last=True)
                val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sub_sampler, drop_last=False)

                model = load_model.load(eeg_enc_name, **constants.model_constants[eeg_enc_name])
                    
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
                if "multimodal" in task:
                    trainer = MultiModalTrainer(model=model, optimizer=optim, n_epochs=epochs,
                                    n_classes=constants.data_constants['n_classes'], save_path=paths['save_path'],
                                    modality=constants.data_constants["modality"], device=device, scheduler=scheduler, warmup=None,
                                    warmup_steps=constants.training_constants['warmup_steps'], batch_size=batch_size)
                else:
                    trainer = ModelTrainer(model=model, optimizer=optim, n_epochs=epochs,
                                        n_classes=constants.data_constants['n_classes'], save_path=paths['save_path'],
                                        weights=None, device=device, scheduler=scheduler, warmup=None,
                                        warmup_steps=constants.training_constants['warmup_steps'], batch_size=batch_size)
                best_model = trainer.train(train_loader, val_loader)

                # best_models_win.append(best_model)
                # best_models.append(best_models_win)
                # best_windows.append(win)
                # median_windows.append(np.median(np.asarray([m['auroc'] for m in best_models_win])))

                test_sub_sampler = torch.utils.data.SubsetRandomSampler(test_index, generator=g)
                test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sub_sampler, drop_last=False)

                new_model = load_model.load(eeg_enc_name, **constants.model_constants[eeg_enc_name])
                new_model = new_model.double()
                new_model.to(device)
                new_model.load_state_dict(best_model['model_state_dict'])
                new_model.eval()
                test_loss, test_acc, test_auroc, y_true_test, y_pred_test = trainer.evaluate(new_model, test_loader)

                metrics['auroc'].append(best_model['auroc'])
                metrics['test_auroc'].append(test_auroc)
                metrics['loss'].append(best_model['loss'])
                metrics['acc'].append(best_model['acc'])
                metrics['epoch'].append(best_model['epoch'])

                print(f"Best Validation AUC = {best_model['auroc']} (epoch = {best_model['epoch']}) |  Best Test AUC = {test_auroc}")

    with open(os.path.join(paths['save_path'], f'{time_windows[0][0]}_{time_windows[0][1]}.pkl'),
            'wb') as f:
        pickle.dump(metrics, f)
    
    print("Median VAL AUC = ", np.median(np.asarray(metrics['auroc'])))
    print("Average VAL AUC = ", np.mean(np.asarray(metrics['auroc'])))

    print("\nMedian TEST AUC = ", np.median(np.asarray(metrics['test_auroc'])))
    print("Average TEST AUC = ", np.mean(np.asarray(metrics['test_auroc'])))

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
    parser.add_argument('--data', type=str, default='ebg4')
    parser.add_argument('--data_type', type=str, default="sensor_ica")
    parser.add_argument('--tmin', type=float, default=None)
    parser.add_argument('--tmax', type=float, default=None)
    parser.add_argument('-w', type=float, default=None)
    parser.add_argument('--ebg_transform', type=str, default='tfr_morlet')
    parser.add_argument('--subject_id', type=int, default=0)
    parser.add_argument('--eeg', type=str, default='eegnet1d')
    parser.add_argument('--model1', type=str, default='resnet1d')
    parser.add_argument('--model2', type=str, default='resnet1d')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=1000)
    # parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--task', type=str, default="whole_win")
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
    data_type = args.data_type
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

    os.makedirs(os.path.join(main_paths['save_path'], args.task), exist_ok=True)
    main_paths['save_path'] = os.path.join(main_paths['save_path'], args.task)

    constants.training_constants['lr'] = args.lr
    constants.training_constants['patience'] = args.patience
    constants.data_constants['tmin'] = args.tmin
    constants.data_constants['tmax'] = args.tmax
    constants.data_constants['w'] = args.w
    constants.data_constants['ebg_transform'] = args.ebg_transform

    directory_name = \
            f"{dataset_name}_{eeg_enc_name}_{constants.data_constants['modality']}"
    os.makedirs(os.path.join(main_paths['save_path'], directory_name), exist_ok=True)

    if constants.data_constants['modality'] == "source":
        n_channels = 4
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 4
    elif constants.data_constants['modality'] == "ebg":
        n_channels = 4
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 4
    elif constants.data_constants['modality'] == "eeg":
        n_channels = 63
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 63
    elif constants.data_constants['modality'] in ["ebg-sniff", "sniff-ebg"]:
        n_channels = 5
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 5
    elif constants.data_constants['modality'] in ["eeg-sniff", "sniff-eeg"]:
        n_channels = 64
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 64
    elif constants.data_constants['modality'] == "both-sniff":
        n_channels = 68
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 68
    elif constants.data_constants['modality'] == 'sniff':
        n_channels = 1
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 1
    elif constants.data_constants['modality'] in ['source-sniff', 'sniff-source']:
        n_channels = 5
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 5
    elif constants.data_constants['modality'] in ['source-ebg', 'ebg-source']:
        n_channels = 8
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 8
    elif constants.data_constants['modality'] in ['source-eeg', 'eeg-source']:
        n_channels = 67
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 67
    elif constants.data_constants['modality'] in ['eeg-ebg', 'ebg-eeg']:
        n_channels = 67
        for key in constants.model_constants.keys():
            if "n_channels" in constants.model_constants[key].keys():
                constants.model_constants[key]['n_channels'] = 67
    else:
        raise NotImplementedError
    # load a sample subject's data to compute the number of time samples
    data_array, labels_array, t, sfreq = load_ebg4(
        root=os.path.join(main_paths['eeg_data'], dataset_name),
        subject_id=1,
        data_type=data_type,
        **constants.data_constants
    )

    if "whole_win" in args.task:
        win = (args.tmin, args.tmax)
    else:
        win = time_windows[0]
    
    data = EBG4(root_path=os.path.join(main_paths['eeg_data'], "ebg4"),
        source_data=data_array, label=labels_array, time_vec=t, fs=sfreq, 
        tmin=win[0], tmax=win[1], w=None, binary=constants.data_constants['binary'],
        data_type=data_type, modality=constants.data_constants["modality"], intensity=constants.data_constants['intensity'],
        pick_subjects=1, fs_new=constants.data_constants['fs_new'], 
        normalize=constants.data_constants['normalize'], transform=None
    )

    n_time_samples = data.data.shape[-1]

    # constants.training_constants['fold'] = args.fold
    constants.model_constants['eegnet']['n_samples'] = n_time_samples
    constants.model_constants['eegnet1d']['n_samples'] = n_time_samples
    constants.model_constants['eegnet_attention']['n_samples'] = n_time_samples
    constants.model_constants['resnet1d']['n_samples'] = n_time_samples
    constants.model_constants['resnet1d']['net_seq_length'][0] = n_time_samples
    constants.model_constants['mlp']['n_samples'] = n_time_samples * n_channels
    if 'multimodal' in args.task:
        constants.model_constants['multimodal']['model1'] = args.model1
        constants.model_constants['multimodal']['model2'] = args.model2
        constants.model_constants['multimodal']['model1_kwargs'] = copy.deepcopy(constants.model_constants[args.model1])
        constants.model_constants['multimodal']['model2_kwargs'] = copy.deepcopy(constants.model_constants[args.model2])
        modality_all = constants.data_constants["modality"]
        modality1, modality2 = modality_all.split('-')
        if "n_channels" in constants.model_constants['multimodal']['model1_kwargs'].keys():
            constants.model_constants['multimodal']['model1_kwargs']['n_channels'] = channel_dict[modality1]
        if "n_channels" in constants.model_constants['multimodal']['model2_kwargs'].keys():
            constants.model_constants['multimodal']['model2_kwargs']['n_channels'] = channel_dict[modality2]
    constants.model_constants['multimodal']['device'] = device
    subject_data_list = [(sid, eeg_enc_name, dataset_name, data_type, epochs, seed, args.task, directory_name, device, constants)
                         for sid in subject_ids]
    print("number of available CPUs = ", os.cpu_count())
    
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
