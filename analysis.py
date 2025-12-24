import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Bạn copy đường dẫn folder kết quả vào đây)
# ==========================================
# Ví dụ: './results/long_term_forecast_ECL_...'
folder_path = '/home/myvh/hoanglmv/Time-LLM/results/long_term_forecast_ECL_512_96_Autoformer_Autoformer_ECL_ftM_sl512_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_Exp_Autoformer_Electricity_0-Autoformer_ECL'

# ==========================================
# 2. LOAD DỮ LIỆU
# ==========================================
pred_path = os.path.join(folder_path, 'pred.npy')
true_path = os.path.join(folder_path, 'true.npy')

if not os.path.exists(pred_path):
    print(f"❌ Không tìm thấy file tại: {folder_path}")
    exit()

preds = np.load(pred_path)
trues = np.load(true_path)

print(f"📦 Shape của dự đoán: {preds.shape}")
print(f"📦 Shape của thực tế: {trues.shape}")
# Thường là (Số lượng mẫu test, Độ dài dự đoán, Số lượng features)
# Ví dụ: (2600, 96, 321)

# ==========================================
# 3. VẼ BIỂU ĐỒ SO SÁNH
# ==========================================
# Chọn mẫu ngẫu nhiên để vẽ
sample_idx = 0  # Chọn mẫu đầu tiên trong tập test
feature_idx = -4 # Chọn đặc trưng cuối cùng (thường là target chính - OT)

# Lấy chuỗi dữ liệu (96 điểm dự đoán)
pred_series = preds[sample_idx, :, feature_idx]
true_series = trues[sample_idx, :, feature_idx]

plt.figure(figsize=(12, 6))
plt.plot(true_series, label='Thực tế (Ground Truth)', color='blue', linewidth=2)
plt.plot(pred_series, label='Dự đoán (Prediction)', color='red', linestyle='--', linewidth=2)

plt.title(f'So sánh kết quả dự đoán (Sample {sample_idx}, Feature {feature_idx})')
plt.xlabel('Thời gian (Time steps)')
plt.ylabel('Giá trị (Normalized Value)')
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 4. TÍNH TOÁN CHỈ SỐ (MSE/MAE)
# ==========================================
mse = np.mean((preds - trues) ** 2)
mae = np.mean(np.abs(preds - trues))

print("="*30)
print(f"📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TOÀN BỘ TẬP TEST:")
print(f"   MSE: {mse:.6f}")
print(f"   MAE: {mae:.6f}")
print("="*30)