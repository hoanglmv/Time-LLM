import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN TỰ ĐỘNG ---
# Giúp chạy file từ bất kỳ đâu (gốc hay utils đều được)
current_script_path = os.path.abspath(__file__)
utils_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(utils_dir)
results_path = os.path.join(project_root, 'results')

print(f"📂 Đang tìm kết quả trong: {results_path}")

# Danh sách tên các đặc trưng KPI (khớp với thứ tự trong file CSV)
feature_labels = [
    "Avg RRC Users", 
    "PRB DL Used", 
    "PRB Available", 
    "PRB Utilization",
    "PS Traffic (Target)" 
]

# Kiểm tra thư mục kết quả
if not os.path.exists(results_path):
    print(f"❌ Không tìm thấy thư mục results tại: {results_path}")
    exit()

list_of_folders = glob.glob(os.path.join(results_path, '*'))
if not list_of_folders:
    print("❌ Thư mục results trống. Hãy chạy Training trước!")
    exit()

# Lấy thư mục kết quả mới nhất
latest_folder = max(list_of_folders, key=os.path.getctime)
print(f"📂 Đang đọc dữ liệu từ: {latest_folder}")

# --- PHẦN 1: VẼ BIỂU ĐỒ LOSS ---
loss_file = os.path.join(latest_folder, 'loss.npy')
if os.path.exists(loss_file):
    print("📈 Đang vẽ biểu đồ Loss...")
    loss_data = np.load(loss_file)
    # loss_data shape: (Epochs, 3) -> [Train, Val, Test]
    
    epochs = range(1, len(loss_data) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_data[:, 0], label='Train Loss', marker='o')
    plt.plot(epochs, loss_data[:, 1], label='Validation Loss', marker='o')
    plt.plot(epochs, loss_data[:, 2], label='Test Loss', marker='o')
    
    plt.title('Hàm Loss qua các Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("⚠️ Không tìm thấy file loss.npy. (Có thể bạn chạy mode Test nên không có Loss mới)")

# --- PHẦN 2: VẼ BIỂU ĐỒ DỰ BÁO VS THỰC TẾ ---
try:
    preds = np.load(os.path.join(latest_folder, 'pred.npy'))
    trues = np.load(os.path.join(latest_folder, 'true.npy'))
    
    print(f"📊 Kích thước dữ liệu Test: {preds.shape}")
    
    # Chọn mẫu đầu tiên trong tập test để vẽ
    sample_idx = 0 
    
    # Tạo lưới biểu đồ (5 dòng, 1 cột)
    fig, axs = plt.subplots(len(feature_labels), 1, figsize=(12, 15), sharex=True)
    
    # Xử lý trường hợp có 1 feature (tránh lỗi vòng lặp)
    if len(feature_labels) == 1: axs = [axs]

    for i in range(len(feature_labels)):
        if i >= trues.shape[2]: break # Tránh lỗi index nếu số feature không khớp
        
        y_true = trues[sample_idx, :, i]
        y_pred = preds[sample_idx, :, i]
        
        axs[i].plot(y_true, label='Thực tế (Ground Truth)', color='blue', linewidth=2)
        axs[i].plot(y_pred, label='Dự báo (Prediction)', color='red', linestyle='--', linewidth=2)
        axs[i].set_title(f"KPI: {feature_labels[i]}")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
        
        if i == len(feature_labels) - 1:
            axs[i].set_xlabel("Thời gian dự báo (Step: 15 phút)")

    plt.tight_layout()
    plt.show()
    print("✅ Đã vẽ xong biểu đồ dự báo!")

except Exception as e:
    print(f"❌ Lỗi khi đọc file dự báo: {e}")