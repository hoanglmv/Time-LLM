# -*- coding: utf-8 -*-
# File này dùng để chạy thử nghiệm một vòng huấn luyện (training loop) nhanh
# với mô hình DLinear để kiểm tra lỗi và đảm bảo các thành phần cốt lõi hoạt động đúng.
# Nó sử dụng một cấu hình tối giản, không dùng command-line arguments.

import torch
from torch import nn, optim
from tqdm import tqdm
import time
import random
import numpy as np
import os
import sys

# Thêm đường dẫn gốc của dự án vào sys.path
# Điều này đảm bảo các module như 'models', 'prepare_data' có thể được import
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models import DLinear
from prepare_data.data_provider.data_factory import data_provider

# --- Cấu hình cho việc chạy nhanh ---
class FastConfig:
    def __init__(self):
        # Basic
        self.is_training = 1
        self.model_id = 'DLinear_fast_test'
        self.model = 'DLinear'
        self.task_name = 'long_term_forecast'
        
        # Data
        self.data = 'ETTh1'
        self.root_path = './dataset/ETT-small/'
        self.data_path = 'ETTh1.csv'
        self.features = 'M' # Multivariate
        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'
        self.percent = 10 # Chỉ dùng 10% dữ liệu

        # Model & Task
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.individual = False # DLinear specific
        self.moving_avg = 25

        # Training
        self.train_epochs = 1
        self.batch_size = 16
        self.learning_rate = 0.001
        self.loss = 'MSE'
        self.num_workers = 0 # 0 để debug dễ hơn trên Windows
        self.embed = 'timeF' # Không quá quan trọng với DLinear nhưng cần cho data_provider
        self.seasonal_patterns= 'Monthly' # Placeholder

def run_fast_test():
    """
    Hàm chính để chạy thử nghiệm nhanh.
    """
    args = FastConfig()
    
    # Thiết lập device (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên thiết bị: {device}")

    # --- Tải dữ liệu ---
    print("\n[1/4] ⏳ Đang tải dữ liệu...")
    try:
        train_data, train_loader = data_provider(args, 'train')
        print(f"✅ Tải dữ liệu thành công. Số lượng batch training: {len(train_loader)}")
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        print("💡 Gợi ý: Bạn đã tải và đặt dataset vào đúng thư mục './dataset/ETT-small/' chưa?")
        return

    # --- Khởi tạo mô hình ---
    print("\n[2/4] ⚙️  Đang khởi tạo mô hình DLinear...")
    model = DLinear.Model(args).float().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    print("✅ Khởi tạo mô hình thành công.")

    # --- Vòng lặp Huấn luyện ---
    print(f"\n[3/4] 🔥 Bắt đầu huấn luyện nhanh cho {args.train_epochs} epoch...")
    model.train()
    epoch_time = time.time()

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        # Đưa dữ liệu lên device
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # Tạo decoder input
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

        # Forward pass
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # Lấy phần output tương ứng với pred_len
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
        
        loss = criterion(outputs, batch_y)
        
        # Backward pass và cập nhật trọng số
        loss.backward()
        optimizer.step()

        if i % 50 == 0: # In loss định kỳ
            print(f"\n   Batch {i}/{len(train_loader)} | Loss: {loss.item():.6f}")

    print(f"✅ Huấn luyện hoàn tất trong {time.time() - epoch_time:.2f} giây.")

    # --- Hoàn tất ---
    print("\n[4/4] 🎉 Chạy thử nghiệm thành công!")
    print("   - Mô hình DLinear đã chạy qua 1 epoch mà không có lỗi runtime.")
    print("   - Pipeline dữ liệu và training hoạt động bình thường.")

if __name__ == '__main__':
    # Thiết lập seed để kết quả có thể tái lập
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    run_fast_test()
