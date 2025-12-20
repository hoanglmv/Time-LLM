# -*- coding: utf-8 -*-
# File này cho phép bạn tùy chỉnh các tham số và chạy thử nghiệm một cách dễ dàng.
# Chỉ cần chỉnh sửa các giá trị trong lớp `Config` dưới đây và chạy file.

import torch
from torch import nn, optim
from tqdm import tqdm
import time
import random
import numpy as np
import os
import sys

# Thêm đường dẫn gốc của dự án vào sys.path để import các module tùy chỉnh
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models import Autoformer, DLinear, TimeLLM
from prepare_data.data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, vali, load_content
from utils.visualize import plot_loss, plot_classification_metrics

# =================================================================================
# =================== KHU VỰC CẤU HÌNH (CONFIGURATION AREA) ===================
# =================================================================================
# Chỉnh sửa các giá trị trong lớp `Config` này để chạy thử nghiệm của bạn.
class Config:
    def __init__(self):
        # --- Cấu hình Thử nghiệm Chính (Thay đổi ở đây) ---
        self.model = 'DLinear'              # Model: 'DLinear', 'Autoformer', 'TimeLLM'
        self.data = 'ETTh1'                 # Dataset: 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic'
        self.train_epochs = 3               # Số epochs để huấn luyện
        self.learning_rate = 0.005          # Tốc độ học (learning rate)
        self.batch_size = 32                # Kích thước batch

        # --- Cấu hình Tác vụ & Mô hình (Thay đổi nếu cần) ---
        self.is_training = 1                # 1: Huấn luyện; 0: Chỉ test
        self.task_name = 'long_term_forecast'
        self.seq_len = 96                   # Độ dài chuỗi đầu vào
        self.pred_len = 96                  # Độ dài chuỗi dự đoán
        self.label_len = 48                 # Độ dài của start token
        
        # --- Cấu hình Dữ liệu (Thường không cần đổi nếu dùng dataset chuẩn) ---
        self.root_path = './dataset/ETT-small/' # Đường dẫn gốc tới thư mục dataset
        self.data_path = 'ETTh1.csv'        # Tên file dữ liệu
        self.features = 'M'                 # 'M', 'S', 'MS'
        self.target = 'OT'                  # Cột mục tiêu
        self.freq = 'h'                     # Tần suất: 'h' (giờ), 't' (phút), 'd' (ngày)
        self.percent = 100                  # Phần trăm dữ liệu sử dụng (10-100)

        # --- Cấu hình Chi tiết Mô hình (Nâng cao) ---
        # Tự động điền một số tham số dựa trên dataset
        self.enc_in, self.dec_in, self.c_out = self.get_dims_based_on_data()
        
        self.d_model = 512                  # Kích thước embedding của model
        self.d_ff = 2048                    # Kích thước lớp Feed-Forward
        self.n_heads = 8                    # Số lượng attention heads
        self.e_layers = 2                   # Số lớp encoder
        self.d_layers = 1                   # Số lớp decoder
        self.dropout = 0.1
        self.moving_avg = 25                # Kích thước cửa sổ cho DLinear và Autoformer
        self.individual = False             # Dành riêng cho DLinear
        self.llm_layers = 6                 # Số lớp LLM cho TimeLLM

        # --- Cấu hình Lưu trữ & Tối ưu hóa ---
        self.model_id = f'{self.model}_{self.data}_sl{self.seq_len}_pl{self.pred_len}'
        self.checkpoints = './checkpoints/'
        self.loss = 'MSE'
        self.lradj = 'type1'
        self.patience = 3
        self.num_workers = 0 # 0 để debug dễ hơn trên Windows
        self.embed = 'timeF'
        self.seasonal_patterns= 'Monthly' # Placeholder cho dataset M4

    def get_dims_based_on_data(self):
        """Tự động trả về số features cho các bộ dữ liệu tiêu chuẩn."""
        data_dims = {
            'ETTh1': (7, 7, 7),
            'ETTh2': (7, 7, 7),
            'ETTm1': (7, 7, 7),
            'ETTm2': (7, 7, 7),
            'Weather': (21, 21, 21),
            'ECL': (321, 321, 321),
            'Traffic': (862, 862, 862),
        }
        return data_dims.get(self.data, (1, 1, 1)) # Mặc định là 1 nếu không tìm thấy


# =================================================================================
# =================== MÃ THỰC THI (EXECUTION CODE) ===================
# =================================================================================
# Bạn thường không cần chỉnh sửa phần code bên dưới.

def run_custom():
    """Hàm chính để chạy thử nghiệm với cấu hình đã định nghĩa."""
    args = Config()
    
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên thiết bị: {device}")
    
    # Thiết lập seed để kết quả có thể tái lập
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # --- Tải dữ liệu ---
    print("\n[1/4] ⏳ Đang tải dữ liệu...")
    try:
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        print(f"✅ Tải dữ liệu '{args.data}' thành công. {len(train_loader)} training batches.")
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        print(f"💡 Gợi ý: Hãy chắc chắn rằng 'root_path' và 'data_path' trong Config là chính xác.")
        return

    # --- Khởi tạo mô hình ---
    print(f"\n[2/4] ⚙️  Đang khởi tạo mô hình {args.model}...")
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float().to(device)
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float().to(device)
    else: # TimeLLM
        model = TimeLLM.Model(args).float().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() if args.task_name != 'classification' else nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    save_path = os.path.join(args.checkpoints, args.model_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"✅ Khởi tạo mô hình thành công. Checkpoint sẽ được lưu tại: {save_path}")

    # --- Huấn luyện ---
    if args.is_training:
        print(f"\n[3/4] 🔥 Bắt đầu huấn luyện cho {args.train_epochs} epochs...")
        
        # Lịch sử để vẽ biểu đồ
        history = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'acc': [],
            'f1': [],
            'precision': [],
            'recall': []
        }

        for epoch in range(args.train_epochs):
            model.train()
            epoch_time = time.time()
            train_loss = []
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.train_epochs}"):
                optimizer.zero_grad()

                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if args.task_name == 'classification':
                    loss = criterion(outputs, batch_y.long().squeeze())
                else:
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            avg_train_loss = np.average(train_loss)
            history['train_loss'].append(avg_train_loss)
            print(f"\nEpoch {epoch + 1} | Time: {time.time() - epoch_time:.2f}s | Train Loss: {avg_train_loss:.6f}")
            
            # Đánh giá trên bộ validation và test
            vali_results = vali(args, None, model, vali_data, vali_loader, criterion, nn.L1Loss())
            test_results = vali(args, None, model, test_data, test_loader, criterion, nn.L1Loss())
            
            history['val_loss'].append(vali_results['loss'])
            history['test_loss'].append(test_results['loss'])

            if args.task_name == 'classification':
                print(f"              Vali Loss: {vali_results['loss']:.4f} | Acc: {vali_results['acc']:.4f} | F1: {vali_results['f1']:.4f}")
                history['acc'].append(vali_results['acc'])
                history['f1'].append(vali_results['f1'])
                history['precision'].append(vali_results['precision'])
                history['recall'].append(vali_results['recall'])
            else:
                print(f"              Vali Loss: {vali_results['loss']:.6f} | Test Loss: {test_results['loss']:.6f}")

            early_stopping(vali_results['loss'], model, save_path)
            if early_stopping.early_stop:
                print("Early stopping!")
                break
            
            adjust_learning_rate(None, optimizer, None, epoch + 1, args, printout=True)
            
        print("✅ Huấn luyện hoàn tất.")
        
        # --- Vẽ và lưu biểu đồ ---
        print("\n[---] 📈 Đang tạo và lưu biểu đồ...")
        fig_save_path = os.path.join('figures', args.model_id)
        plot_loss(history['train_loss'], history['val_loss'], save_path=f"{fig_save_path}_loss.png")
        if args.task_name == 'classification':
            plot_classification_metrics({k: v for k, v in history.items() if k in ['acc', 'f1', 'precision', 'recall']},
                                        save_path=f"{fig_save_path}_metrics.png")

    # --- Đánh giá cuối cùng ---
    print("\n[4/4] 📊 Đang tải model tốt nhất và thực hiện đánh giá cuối cùng...")
    best_model_path = os.path.join(save_path, 'checkpoint.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("✅ Tải model tốt nhất thành công.")
        final_results = vali(args, None, model, test_data, test_loader, criterion, nn.L1Loss())
        
        if args.task_name == 'classification':
            print(f"🎉 Hoàn tất! Kết quả cuối cùng trên bộ test -> Loss: {final_results['loss']:.4f} | Acc: {final_results['acc']:.4f} | F1: {final_results['f1']:.4f}")
        else:
            print(f"🎉 Hoàn tất! Kết quả cuối cùng trên bộ test -> Loss (MSE): {final_results['loss']:.6f} | MAE: {final_results['mae']:.6f}")
    else:
        print(f"❌ Không tìm thấy checkpoint tại {best_model_path}. Không thể thực hiện đánh giá cuối cùng.")

if __name__ == '__main__':
    run_custom()
