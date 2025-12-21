# Copyright 2024 The Time-LLM Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
# Add the project root to sys.path for local module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import pandas as pd # <--- ĐÃ THÊM THƯ VIỆN PANDAS

from models import Autoformer, DLinear, TimeLLM
from prepare_data.data_provider.data_factory import data_provider
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from utils.visualize import plot_loss, plot_classification_metrics

# Thiết lập môi trường
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S: univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')
parser.add_argument('--llm_model_id', type=str, default='gpt2', help='LLM model id')
parser.add_argument('--llm_dim', type=str, default='4096', help='LLM model dimension')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

if __name__ == '__main__':
    args = parser.parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Khởi tạo Accelerator (Bỏ qua DeepSpeed để tránh lỗi Windows)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # --- IN THÔNG TIN THIẾT BỊ ---
    print("\n" + "="*40)
    if accelerator.device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(accelerator.device)
        vram = torch.cuda.get_device_properties(accelerator.device).total_memory / 1e9
        print(f"🚀 Đang chạy trên GPU: {gpu_name}")
        print(f"📦 VRAM khả dụng: {vram:.2f} GB")
    else:
        print(f"⚠️ Đang chạy trên CPU (Sẽ rất chậm!)")
    print("="*40 + "\n")
    # -----------------------------

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.factor,
            args.embed, args.des, ii)

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            model = TimeLLM.Model(args).float()

        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        args.content = load_content(args)

        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.CrossEntropyLoss() if args.task_name == 'classification' else nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # === PHẦN TRAINING (Chỉ chạy khi --is_training 1) ===
        if args.is_training:
            accelerator.print(">>>>>>> Start Training <<<<<<<")
            # Lịch sử để vẽ biểu đồ
            history = {
                'train_loss': [], 'val_loss': [], 'test_loss': [],
                'acc': [], 'f1': [], 'precision': [], 'recall': []
            }

            for epoch in range(args.train_epochs):
                iter_count = 0
                train_loss = []
                model.train()
                epoch_time = time.time()

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.float().to(accelerator.device)
                    batch_y = batch_y.float().to(accelerator.device)
                    batch_x_mark = batch_x_mark.float().to(accelerator.device)
                    batch_y_mark = batch_y_mark.float().to(accelerator.device)

                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

                    # Forward pass
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # Tính loss dựa trên tác vụ
                    if args.task_name == 'classification':
                        loss = criterion(outputs, batch_y.long().squeeze())
                    else:
                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        accelerator.print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                        iter_count = 0
                        time_now = time.time()

                    accelerator.backward(loss)
                    model_optim.step()

                # Kết thúc Epoch
                accelerator.print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
                avg_train_loss = np.average(train_loss)
                history['train_loss'].append(avg_train_loss)

                # Đánh giá
                vali_results = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
                test_results = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)

                history['val_loss'].append(vali_results['loss'])
                history['test_loss'].append(test_results['loss'])

                # In kết quả
                if args.task_name == 'classification':
                    accelerator.print(f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Vali Loss: {vali_results['loss']:.4f} | Acc: {vali_results['acc']:.4f} | F1: {vali_results['f1']:.4f}")
                    history['acc'].append(vali_results['acc'])
                    history['f1'].append(vali_results['f1'])
                    history['precision'].append(vali_results['precision'])
                    history['recall'].append(vali_results['recall'])
                else:
                    accelerator.print(f"Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Vali Loss: {vali_results['loss']:.6f} | Test Loss: {test_results['loss']:.6f}")

                early_stopping(vali_results['loss'], model, path)
                if early_stopping.early_stop:
                    accelerator.print("Early stopping")
                    break

                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            accelerator.print(">>>>>>> Training Finished <<<<<<<")

            # === PHẦN THÊM MỚI: LƯU LOSS HISTORY RA CSV ===
            if accelerator.is_local_main_process:
                try:
                    # Tạo dictionary chứa dữ liệu
                    loss_data = {
                        'Epoch': list(range(1, len(history['train_loss']) + 1)),
                        'Train Loss': history['train_loss'],
                        'Val Loss': history['val_loss'],
                        'Test Loss': history['test_loss']
                    }
                    
                    # Thêm các metrics khác nếu là bài toán phân loại
                    if args.task_name == 'classification':
                        for key in ['acc', 'f1', 'precision', 'recall']:
                            if key in history and len(history[key]) > 0:
                                loss_data[key] = history[key]

                    # Tạo DataFrame và lưu
                    df_loss = pd.DataFrame(loss_data)
                    loss_save_path = os.path.join(path, 'loss_history.csv')
                    df_loss.to_csv(loss_save_path, index=False)
                    accelerator.print(f"✅ Đã lưu file lịch sử Loss tại: {loss_save_path}")
                except Exception as e:
                    accelerator.print(f"❌ Lỗi khi lưu file CSV loss: {e}")
            # ===============================================

            # Vẽ và lưu biểu đồ
            if accelerator.is_local_main_process:
                fig_save_path = os.path.join('figures', setting + '-' + args.model_comment)
                # Đảm bảo thư mục figures tồn tại
                if not os.path.exists('figures'):
                    os.makedirs('figures')
                    
                plot_loss(history['train_loss'], history['val_loss'], save_path=f"{fig_save_path}_loss.png")
                if args.task_name == 'classification':
                    plot_classification_metrics({k: v for k, v in history.items() if k in ['acc', 'f1', 'precision', 'recall']},
                                                save_path=f"{fig_save_path}_metrics.png")

        # === PHẦN TESTING & PREDICTION (Chạy cho cả Train và Test mode) ===
        accelerator.print('>>>>>>> Loading Best Model for Testing <<<<<<<')
        best_model_path = path + '/' + 'checkpoint.pth'

        if not os.path.exists(best_model_path):
            accelerator.print(f"❌ Không tìm thấy file checkpoint tại: {best_model_path}")
        else:
            # Tải model đã lưu
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=accelerator.device))

            folder_path = './results/' + setting + '-' + args.model_comment + '/'
            os.makedirs(folder_path, exist_ok=True)

            accelerator.print('>>>>>>> Start Final Evaluation <<<<<<<')
            final_results = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric, folder_path=folder_path)

            if args.task_name == 'classification':
                accelerator.print(f"🎉 Final Test Results -> Loss: {final_results['loss']:.4f} | Acc: {final_results['acc']:.4f} | F1: {final_results['f1']:.4f}")
            else:
                accelerator.print(f"🎉 Final Test Results -> Loss(MSE): {final_results['loss']:.6f} | MAE: {final_results['mae']:.6f}")
                accelerator.print(f'✅ Kết quả dự đoán (pred.npy, true.npy) đã lưu tại: {folder_path}')

    accelerator.wait_for_everyone()

    # TẮT TÍNH NĂNG TỰ XÓA FILE ĐỂ BẢO VỆ MODEL
    # if accelerator.is_local_main_process:
    #     path = './checkpoints'  # unique checkpoint saving path
    #     del_files(path)  # delete checkpoint files
    #     accelerator.print('success delete checkpoints')