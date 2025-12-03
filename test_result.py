import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- CẤU HÌNH ---
# 1. Đường dẫn thư mục results
results_path = './results/'

# 2. Danh sách tên các đặc trưng (Features) theo thứ tự trong file CSV của bạn
# Đảm bảo thứ tự này khớp với file enodebF121.csv sau khi đã bỏ cột date
feature_names = [
    "PS Traffic (MB)",              # Index 0
    "Avg RRC Connected Users",      # Index 1
    "PRB DL Used",                  # Index 2
    "PRB DL Available",             # Index 3
    "PRB Utilization (%)"           # Index 4
]

# 3. Chọn đặc trưng muốn vẽ (Ví dụ: 0 là Traffic, 4 là PRB Util)
feature_index_to_plot = 0 

# 4. Chọn mẫu test muốn xem (0 là mẫu đầu tiên trong tập test)
sample_index = 0
# ----------------

# --- TỰ ĐỘNG TÌM KẾT QUẢ MỚI NHẤT ---
list_of_folders = glob.glob(os.path.join(results_path, '*'))
if not list_of_folders:
    print("❌ Lỗi: Không tìm thấy thư mục kết quả nào trong ./results/")
else:
    # Lấy folder mới nhất dựa trên thời gian tạo
    latest_folder = max(list_of_folders, key=os.path.getctime)
    print(f"📂 Đang đọc kết quả từ thư mục: {latest_folder}")

    try:
        # Load dữ liệu npy
        preds = np.load(os.path.join(latest_folder, 'pred.npy'))
        trues = np.load(os.path.join(latest_folder, 'true.npy'))

        # Shape thường là: (Số lượng mẫu Test, Độ dài dự báo, Số lượng Features)
        # Ví dụ: (200, 96, 5)
        print(f"📊 Kích thước dữ liệu Test: {preds.shape}")

        # --- VẼ BIỂU ĐỒ ---
        plt.figure(figsize=(15, 6))

        # Lấy dữ liệu của mẫu sample_index, tại cột feature_index_to_plot
        y_true = trues[sample_index, :, feature_index_to_plot]
        y_pred = preds[sample_index, :, feature_index_to_plot]

        # Trục X là thời gian (Tương lai)
        x_axis = range(len(y_true))

        plt.plot(x_axis, y_true, label='Thực tế (Ground Truth)', color='blue', linewidth=2)
        plt.plot(x_axis, y_pred, label='Dự báo (Prediction)', color='red', linestyle='--', linewidth=2)

        # Trang trí
        feat_name = feature_names[feature_index_to_plot] if feature_index_to_plot < len(feature_names) else f"Feature {feature_index_to_plot}"
        plt.title(f"So sánh Thực tế vs Dự báo: {feat_name}", fontsize=16)
        plt.xlabel("Thời gian dự báo (Mỗi điểm = 15 phút)", fontsize=12)
        plt.ylabel("Giá trị (Scaled)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Hiển thị
        plt.show()

    except Exception as e:
        print(f"❌ Có lỗi khi đọc file hoặc vẽ biểu đồ: {e}")