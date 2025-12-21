import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# ================= CẤU HÌNH =================
# Bạn hãy copy đường dẫn folder kết quả vừa chạy xong vào đây
# Ví dụ: './results/long_term_forecast_...'
RESULT_FOLDER = './results/long_term_forecast_ECL_512_96_TimeLLM_ECL_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df64_fc3_eb8_TimeLLM-ECL_0'

# ================= XỬ LÝ =================

def visualize_prediction(folder_path, sample_id=0, feature_id=-1):
    """
    sample_id: Chọn mẫu thứ mấy trong tập test để vẽ (0 là mẫu đầu tiên)
    feature_id: Chọn cột feature nào để vẽ (-1 là cột cuối cùng/target)
    """
    print(f"📂 Đang đọc dữ liệu từ: {folder_path}")
    
    try:
        preds = np.load(os.path.join(folder_path, 'pred.npy'))
        trues = np.load(os.path.join(folder_path, 'true.npy'))
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file pred.npy hoặc true.npy. Hãy kiểm tra lại đường dẫn RESULT_FOLDER.")
        return

    print(f"📊 Kích thước tập Test: {preds.shape}")
    # Shape thường là: (Số mẫu, Độ dài dự báo, Số Features)
    
    # Lấy dữ liệu của mẫu được chọn
    # preds[sample_id, :, feature_id] nghĩa là: Lấy mẫu số sample_id, lấy toàn bộ thời gian, lấy feature_id
    pred_curve = preds[sample_id, :, feature_id]
    true_curve = trues[sample_id, :, feature_id]

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(true_curve, label='Thực tế (Ground Truth)', color='blue', linewidth=2)
    plt.plot(pred_curve, label='Dự đoán (Prediction)', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'So sánh Dự đoán vs Thực tế (Sample {sample_id})', fontsize=16)
    plt.xlabel('Thời gian (Time Steps)', fontsize=12)
    plt.ylabel('Giá trị (Normalized)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_loss_history(checkpoint_path):
    """
    Vẽ lại biểu đồ Loss từ file CSV đã lưu ở Bước 1
    """
    csv_path = os.path.join(checkpoint_path, 'loss_history.csv')
    if not os.path.exists(csv_path):
        print("⚠️ Không tìm thấy file loss_history.csv (Có thể bạn chưa thêm code lưu CSV?)")
        return
        
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
    plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss')
    plt.title("Quá trình hội tụ của hàm Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # 1. Vẽ so sánh dự đoán
    visualize_prediction(RESULT_FOLDER, sample_id=0, feature_id=-1)
    
    # Mẹo: Bạn có thể đổi sample_id=10, 20... để xem các mẫu khác nhau
    # visualize_prediction(RESULT_FOLDER, sample_id=20, feature_id=-1)
    
    # 2. (Tùy chọn) Vẽ lại Loss nếu bạn biết đường dẫn checkpoint
    # plot_loss_history('./checkpoints/...')