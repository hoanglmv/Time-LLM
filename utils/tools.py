import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm
import torchmetrics

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, folder_path=None):
    total_loss = []
    total_mae_loss = []
    
    device = accelerator.device if accelerator is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.task_name == 'classification':
        num_classes = vali_data.num_classes
        acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
        precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
        recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)

    preds_list = []
    trues_list = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader), disable=True):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if accelerator is not None:
                outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            if args.task_name == 'classification':
                pred_labels = torch.argmax(outputs, dim=1)
                true_labels = batch_y.long().squeeze().to(device)
                
                acc_metric.update(pred_labels, true_labels)
                f1_metric.update(pred_labels, true_labels)
                precision_metric.update(pred_labels, true_labels)
                recall_metric.update(pred_labels, true_labels)
                
                loss = criterion(outputs, true_labels)
                total_loss.append(loss.item())

            else:
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                mae_loss = mae_metric(pred, true)

                total_loss.append(loss.item())
                total_mae_loss.append(mae_loss.item())
                
                if folder_path is not None:
                    preds_list.append(pred.cpu().numpy())
                    trues_list.append(true.cpu().numpy())

    results = {}
    if args.task_name == 'classification':
        results['acc'] = acc_metric.compute().item()
        results['f1'] = f1_metric.compute().item()
        results['precision'] = precision_metric.compute().item()
        results['recall'] = recall_metric.compute().item()
        results['loss'] = np.average(total_loss)
        
        acc_metric.reset()
        f1_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
    else:
        results['loss'] = np.average(total_loss)
        results['mae'] = np.average(total_mae_loss)

    if folder_path is not None and args.task_name != 'classification':
        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

    model.train()
    return results


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r', encoding='utf-8') as f:
        content = f.read()
    return content