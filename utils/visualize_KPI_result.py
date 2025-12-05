import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN ---
current_script_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(utils_dir)
results_path = os.path.join(project_root, 'results')

# --- CẤU HÌNH NHÃN (SỬA LẠI CHO KHỚP DỮ LIỆU) ---
# Dựa trên logic của data_loader: Target (ps_traffic) bị đẩy xuống cuối.
# Thứ tự còn lại giữ nguyên.
feature_labels = [
    "Avg RRC Users",          # Index 0
    "PRB DL Used",            # Index 1
    "PRB Available",          # Index 2
    "PRB Utilization",        # Index 3
    "PS Traffic (Target)"     # Index 4 (Target luôn ở cuối)
]
# ------------------------------------------------

print(f"📂 Đang tìm kết quả trong: {results_path}")

if not os.path.exists(results_path):
    print(f"❌ Không tìm thấy thư mục results.")
    exit()

list_of_folders = glob.glob(os.path.join(results_path, '*'))
if not list_of_folders:
    print("❌ Thư mục results trống.")
    exit()

# Lấy folder mới nhất
latest_folder = max(list_of_folders, key=os.path.getctime)
print(f"📂 Đang đọc dữ liệu từ: {os.path.basename(latest_folder)}")

try:
    preds = np.load(os.path.join(latest_folder, 'pred.npy'))
    trues = np.load(os.path.join(latest_folder, 'true.npy'))

    print(f"📊 Kích thước dữ liệu dự báo (Shape): {preds.shape}")
    # Shape thường là: (Số mẫu, 96, Số Features)
    
    num_features_data = preds.shape[2]
    print(f"👉 Số lượng Features thực tế trong file npy: {num_features_data}")
    print(f"👉 Số lượng Nhãn bạn khai báo: {len(feature_labels)}")

    if num_features_data != len(feature_labels):
        print("⚠️ CẢNH BÁO: Số lượng features không khớp! Biểu đồ có thể bị lệch nhãn.")

    # --- VẼ BIỂU ĐỒ ---
    sample_idx = 0 
    
    # Tự động điều chỉnh số lượng biểu đồ dựa trên dữ liệu thực tế
    fig, axs = plt.subplots(num_features_data, 1, figsize=(12, 3 * num_features_data), sharex=True)
    if num_features_data == 1: axs = [axs]

    for i in range(num_features_data):
        y_true = trues[sample_idx, :, i]
        y_pred = preds[sample_idx, :, i]
        
        # Lấy nhãn tương ứng (hoặc để mặc định nếu thiếu nhãn)
        label_name = feature_labels[i] if i < len(feature_labels) else f"Feature {i}"

        axs[i].plot(y_true, label='Thực tế', color='blue', linewidth=2)
        axs[i].plot(y_pred, label='Dự báo', color='red', linestyle='--', linewidth=2)
        axs[i].set_title(f"KPI: {label_name} (Index {i})")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
        
        if i == num_features_data - 1:
            axs[i].set_xlabel("Thời gian dự báo (Step: 15 phút)")

    plt.tight_layout()
    plt.show()
    print("✅ Đã vẽ xong!")

except Exception as e:
    print(f"❌ Lỗi: {e}")