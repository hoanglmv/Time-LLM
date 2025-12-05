import numpy as np
import os
import glob

# Tìm folder kết quả mới nhất
results_path = './results/'
list_of_folders = glob.glob(os.path.join(results_path, '*'))
latest_folder = max(list_of_folders, key=os.path.getctime)
print(f"📂 Đang kiểm tra folder: {latest_folder}")

# Load file dự báo
pred = np.load(os.path.join(latest_folder, 'pred.npy'))

print("-" * 30)
print(f"📊 Kích thước (Shape) của pred.npy: {pred.shape}")
print("-" * 30)

# Phân tích kết quả
batch, seq_len, features = pred.shape
if features == 1:
    print("❌ LỖI: File chỉ có 1 feature (c_out=1).")
    print("-> Nguyên nhân: Có thể do file CSV đầu vào bị thiếu cột, hoặc chạy nhầm --features MS")
elif features == 5:
    print("✅ TỐT: File đã có đủ 5 features.")
    print("-> Nguyên nhân: Do file code vẽ biểu đồ (visualize) bị sai danh sách nhãn.")