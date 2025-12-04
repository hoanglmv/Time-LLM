import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- CẤU HÌNH ---
results_path = './results/'

# Tên các chỉ số KPI theo thứ tự trong file CSV của bạn
# Lưu ý: Do bạn dùng --target ps_traffic_mb, cột này thường được code chuyển xuống cuối cùng.
# Tuy nhiên với features='M', thứ tự thường giữ nguyên hoặc đảo nhẹ. 
# Ta cứ đặt tên tạm, quan trọng là nhìn hình dáng đồ thị.
feature_labels = [
    "Avg RRC Users", 
    "PRB DL Used", 
    "PRB Available", 
    "PRB Utilization",
    "PS Traffic (Target)" # Target thường bị đẩy xuống cuối
]

# --- TỰ ĐỘNG TÌM KẾT QUẢ MỚI NHẤT ---
list_of_folders = glob.glob(os.path.join(results_path, '*'))
if not list_of_folders:
    print("❌ Chưa tìm thấy kết quả. Bạn đã chạy lệnh với --is_training 0 chưa?")
    exit()

# Lấy thư mục mới nhất vừa chạy xong
latest_folder = max(list_of_folders, key=os.path.getctime)
print(f"📂 Đang đọc kết quả từ: {latest_folder}")

try:
    # Load dữ liệu
    preds = np.load(os.path.join(latest_folder, 'pred.npy'))
    trues = np.load(os.path.join(latest_folder, 'true.npy'))

    # Shape: (Số mẫu test, Độ dài dự báo 96, Số features 5)
    print(f"📊 Shape dữ liệu: {preds.shape}")

    # --- VẼ BIỂU ĐỒ ---
    # Chọn một mẫu ngẫu nhiên trong tập test để xem (ví dụ mẫu thứ 0)
    sample_idx = 0 
    
    # Tạo 5 biểu đồ con cho 5 chỉ số
    fig, axs = plt.subplots(len(feature_labels), 1, figsize=(12, 15), sharex=True)
    
    for i in range(len(feature_labels)):
        # Lấy dữ liệu của feature thứ i
        y_true = trues[sample_idx, :, i]
        y_pred = preds[sample_idx, :, i]
        
        # Vẽ
        axs[i].plot(y_true, label='Thực tế (Ground Truth)', color='blue', linewidth=2)
        axs[i].plot(y_pred, label='Dự báo (Prediction)', color='red', linestyle='--', linewidth=2)
        axs[i].set_title(f"KPI: {feature_labels[i]}")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
        
        if i == len(feature_labels) - 1:
            axs[i].set_xlabel("Thời gian dự báo (Step: 15 phút)")

    plt.tight_layout()
    plt.show()
    print("✅ Đã vẽ xong biểu đồ!")

except Exception as e:
    print(f"❌ Có lỗi khi đọc file: {e}")