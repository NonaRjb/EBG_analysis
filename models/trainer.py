import torch
from torch import nn
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os


class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            weights: torch.Tensor,
            n_epochs: int,
            n_classes: int,
            scheduler,
            save_path, device='cuda'
    ):

        self.device = device
        self.n_classes = n_classes
        self.model = model.to(device)
        if n_classes == 2:
            self.loss_cls = nn.BCEWithLogitsLoss().to(self.device)
            self.accuracy = Accuracy(task='binary').to(self.device)
            self.auroc = AUROC(task="binary")
        else:
            self.loss_cls = nn.CrossEntropyLoss().to(self.device)
            self.accuracy = Accuracy(task='multiclass', num_classes=n_classes, top_k=1).to(self.device)
            self.auroc = AUROC(task="multiclass", num_classes=n_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = n_epochs
        self.mixed_precision = True

        self.save_path = save_path
        self.patience = 2000
        self.log_freq = 1

        # y_true and y_pred for debugging
        self.y_true_train = []
        self.y_pred_train = []
        self.y_true_val = []
        self.y_pred_val = []

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        scaler = GradScaler(enabled=self.mixed_precision)

        best_model = None
        best_loss = 10000000
        best_acc = 0.0
        patience = self.patience
        print("Training Started...")
        for epoch in range(self.epochs):
            # print(f"Epoch {epoch}/{self.epochs}.")
            steps = 0
            loss_epoch = []
            y_true = []
            y_pred = []
            self.model.train()
            progress_bar = tqdm(train_data_loader, disable=True)
            for x, y in progress_bar:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                if self.n_classes == 2:
                    y = y.to(self.device)
                else:
                    y = y.to(torch.long).to(self.device)

                with autocast(enabled=self.mixed_precision):
                    preds = self.model(x)
                    if len(preds.shape) > 1:
                        preds = preds.squeeze(dim=1)
                    if self.n_classes == 2:
                        loss_classification = self.loss_cls(preds, y.float())
                    else:
                        loss_classification = self.loss_cls(preds.float(), y)

                loss_epoch.append(loss_classification.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds))

                scaler.scale(loss_classification).backward()
                # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                steps += 1

            with torch.no_grad():
                # train_avg_acc = self.accuracy(torch.stack(y_pred), torch.stack(y_true))
                train_loss = np.mean(loss_epoch)
                if self.n_classes == 2:
                    y_pred_bin = torch.stack(y_pred) >= 0.5
                    train_balanced_acc = balanced_accuracy_score(torch.stack(y_true).detach().cpu(), y_pred_bin.
                                                                 detach().cpu())
                else:
                    train_balanced_acc = self.accuracy(torch.stack(y_pred).detach().cpu().float(),
                                                       torch.stack(y_true).detach().cpu())
                train_auroc = self.auroc(torch.stack(y_pred).detach().cpu().float(), torch.stack(y_true).detach().cpu())

            if epoch % self.log_freq == 0 or epoch == self.epochs - 1:
                val_loss, val_balanaced_acc, val_auroc, y_true_val, y_pred_val = self.evaluate(self.model, val_data_loader)
                # if val_loss < best_loss:
                if val_auroc > best_acc:
                    # best_loss = val_loss
                    best_acc = val_auroc
                    patience = self.patience
                    best_model = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                        'acc': val_balanaced_acc,
                        'auroc': val_auroc
                    }
                    # self.y_true_train = torch.stack(y_true).detach().cpu().numpy()
                    # self.y_pred_train = torch.stack(y_pred).detach().cpu()
                    # self.y_true_val = torch.stack(y_true_val).detach().cpu()
                    # self.y_pred_val = torch.stack(y_pred_val).detach().cpu()
                else:
                    patience -= 1

                for param_group in self.optimizer.param_groups:
                    learning_rate = param_group["lr"]

                self.scheduler.step(val_loss)

                # if patience == 0:
                #     break
                print(f'---------------------- Epoch: {epoch} ----------------------')
                print(f'Training AUC: {train_auroc} | Acc.: {train_balanced_acc} | Loss: {train_loss}')
                print(f'Validation AUC: {val_auroc} | Acc: {val_balanaced_acc} | Loss: {val_loss}')
                # print(f'lr = {learning_rate}')
                
                wandb.log({
                    "train_loss": train_loss,
                    "train_auroc": train_auroc,
                    "val_loss": val_loss,
                    "val_auroc": val_auroc,
                    "lr": learning_rate,
                    "epoch": epoch
                })

        print(f"Best Validation AUC Score = {best_model['auroc']} (Epoch = {best_model['epoch']})")
        # if best_model is None:
        # set the best model to the final model
        best_model = {
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss,
            'acc': val_balanaced_acc,
            'auroc': val_auroc
        }

        print(f"Final Validation AUC Score = {best_model['auroc']} (Epoch = {best_model['epoch']})")
        # filename = os.path.join(self.save_path, 'checkpoint.pth.tar')
        # torch.save(best_model, filename)
        # np.save(os.path.join(self.save_path, "y_true_train.npy"), np.array(self.y_true_train))
        # np.save(os.path.join(self.save_path, "y_pred_train.npy"), np.array(self.y_pred_train))
        # np.save(os.path.join(self.save_path, "y_true_val.npy"), np.array(self.y_true_val))
        # np.save(os.path.join(self.save_path, "y_pred_val.npy"), np.array(self.y_pred_val))
        # print("Finished creating checkpoint.")

        # predictions = {
        #     'y_true_train': np.array(self.y_true_train),
        #     'y_pred_train': np.array(self.y_pred_train),
        #     'y_true_val': np.array(self.y_true_val),
        #     'y_pred_val': np.array(self.y_pred_val)
        # }

        return best_model

    def evaluate(self, model, data_loader):

        model.eval()
        y_true = []
        y_pred = []
        loss_epoch = []
        progress_bar = tqdm(data_loader, disable=True)
        with torch.no_grad():
            for x, y in progress_bar:
                x = x.to(self.device)
                if self.n_classes == 2:
                    y = y.to(self.device)
                else:
                    y = y.to(torch.long).to(self.device)

                # with autocast(enabled=self.mixed_precision):
                preds = model(x)
                if len(preds.shape) > 1:
                    preds = preds.squeeze(dim=1)
                if self.n_classes == 2:
                    loss_classification = self.loss_cls(preds, y.float())
                else:
                    loss_classification = self.loss_cls(preds.float(), y)

                loss_epoch.append(loss_classification.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds))

            mean_loss_epoch = np.mean(loss_epoch)
            if self.n_classes == 2:
                y_pred_bin = torch.stack(y_pred) >= 0.5
                avg_acc = balanced_accuracy_score(torch.stack(y_true).detach().cpu(), y_pred_bin.
                                                             detach().cpu())
            else:
                avg_acc = self.accuracy(torch.stack(y_pred).detach().cpu().float(),
                                                   torch.stack(y_true).detach().cpu())
            # avg_acc = self.accuracy(torch.stack(y_pred), torch.stack(y_true))
            auroc = self.auroc(torch.stack(y_pred).detach().cpu().float(), torch.stack(y_true).detach().cpu())

        return mean_loss_epoch, avg_acc, auroc, y_true, y_pred
