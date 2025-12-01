import os
import subprocess
import sys

# ================= CẤU HÌNH NHANH =================
# Giảm Batch size xuống thấp để tránh lỗi tràn RAM (OOM)
BATCH_SIZE = 8  
# Chạy thử 1 epoch để xem có lỗi không (Sau này sửa thành 10)
TRAIN_EPOCHS = 1 
# Độ dài chuỗi đầu vào và dự báo
SEQ_LEN = 96
PRED_LEN = 96
# ==================================================

def check_files():
    """Kiểm tra xem dữ liệu có tồn tại không"""
    data_path = "./dataset/weather/weather.csv"
    if not os.path.exists(data_path):
        print(f"❌ LỖI: Không tìm thấy file dữ liệu tại: {data_path}")
        print("👉 Hãy tạo thư mục 'dataset/weather' và copy file 'weather.csv' vào đó.")
        sys.exit(1)
    else:
        print(f"✅ Đã tìm thấy dữ liệu: {data_path}")

def run_training():
    """Cấu hình lệnh chạy"""
    
    # Danh sách tham số (Arguments)
    # Chúng ta dùng python -u để log hiện ra ngay lập tức
    cmd = [
        sys.executable, "-u", "run_main.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/weather/",
        "--data_path", "weather.csv",
        "--model_id", f"weather_{SEQ_LEN}_{PRED_LEN}",
        "--model", "TimeLLM",
        "--data", "Weather",
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--label_len", str(int(SEQ_LEN/2)), # Thường bằng 1/2 seq_len
        "--pred_len", str(PRED_LEN),
        "--enc_in", "21",
        "--c_out", "21",
        "--des", "Exp_Weather",
        "--itr", "1",
        "--d_model", "32",      # Giảm kích thước model để chạy nhẹ hơn
        "--d_ff", "128",        # Giảm kích thước feed forward
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", "0.001",
        "--llm_layers", "6",    # Số lớp LLaMA (giữ nguyên hoặc giảm xuống 3-4 nếu yếu)
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", "3",
        "--llm_model", "LLAMA", # Hoặc "GPT2" nếu bạn muốn test siêu nhẹ
        "--llm_dim", "4096",    # Dimension của LLaMA-7B
        "--prompt_domain", "1"  # Bật chế độ prompt chuyên dụng cho TimeSeries
    ]

    print("\n🚀 Đang khởi động quá trình huấn luyện...")
    print(f"⚙️  Lệnh thực thi: {' '.join(cmd)}\n")
    print("----------------------------------------------------------------")
    
    try:
        # Chạy lệnh và stream log ra màn hình
        subprocess.run(cmd, check=True)
        print("\n🎉 CHÚC MỪNG! Quá trình chạy thử đã hoàn tất thành công.")
        print("📁 Kiểm tra thư mục './checkpoints' và './results' để xem kết quả.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ QUÁ TRÌNH CHẠY GẶP LỖI (Mã lỗi: {e.returncode})")
        print("👉 Hãy kiểm tra lại log phía trên để xem chi tiết.")
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng thủ công.")

if __name__ == "__main__":
    check_files()
    run_training()