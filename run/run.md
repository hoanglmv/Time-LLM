# Hướng dẫn Chạy và Cấu hình các kịch bản

Tài liệu này hướng dẫn cách sử dụng các kịch bản (scripts) khác nhau trong dự án Time-LLM để huấn luyện, tiền huấn luyện và đánh giá mô hình.

## Cấu trúc thư mục

- `run/`: Chứa các file Python chính để thực thi (`run_main.py`, `run_pretrain.py`, `run_m4.py`, `run_fast.py`).
- `scripts/`: Chứa các file shell (`.sh`) đã được cấu hình sẵn để chạy các kịch bản trên với các bộ tham số khác nhau.
    - `scripts/single_gpu/`: Kịch bản cho việc chạy trên một GPU.
    - `scripts/multi_gpu/`: Kịch bản cho việc chạy trên nhiều GPU sử dụng `accelerate`.

---

## 1. Cách chạy nhanh để kiểm tra (Quick Test Run)

Để kiểm tra nhanh môi trường và pipeline xem có lỗi không, bạn có thể sử dụng `run_fast.py`. File này được thiết kế để chạy một mô hình DLinear trên một phần nhỏ của bộ dữ liệu ETTh1 trong 1 epoch.

**Lệnh thực thi:**

(Chạy từ thư mục gốc của dự án)
```bash
python run/run_fast.py
```

Nếu kịch bản này chạy thành công, có nghĩa là môi trường của bạn đã được cài đặt đúng và các thành phần cốt lõi đang hoạt động.

---

## 2. Cách chạy Training & Evaluation chính thức

Cách được khuyến nghị để huấn luyện và đánh giá mô hình là sử dụng các kịch bản shell có sẵn trong thư mục `scripts/`. Các kịch bản này đã được thiết lập với các bộ tham số đã được kiểm chứng cho từng bộ dữ liệu.

**Quan trọng:** Luôn chạy các lệnh sau từ thư mục gốc của dự án (`E:\Project\Time-LLM`).

### Chạy trên một GPU (Single GPU)

Các kịch bản trong `scripts/single_gpu/` được cấu hình để chạy trên GPU đầu tiên (`CUDA_VISIBLE_DEVICES=0`).

**Ví dụ:** Chạy training cho mô hình TimeLLM trên bộ dữ liệu ETTh1.
```bash
# Đối với người dùng Linux/macOS hoặc Git Bash/WSL trên Windows
sh scripts/single_gpu/TimeLLM_ETTh1.sh
```

Bạn có thể mở file `.sh` để chỉnh sửa các tham số như `train_epochs`, `learning_rate` nếu cần.

### Chạy trên nhiều GPU (Multi GPU)

Các kịch bản này yêu cầu `accelerate` của Hugging Face.

**Ví dụ:** Chạy training cho mô hình TimeLLM trên bộ dữ liệu ETTh1 với nhiều GPU.
```bash
accelerate launch scripts/multi_gpu/TimeLLM_ETTh1.sh
```
`accelerate` sẽ tự động quản lý việc phân phối quá trình huấn luyện trên các GPU có sẵn.

---

## 3. Giải thích các tham số chính

Bạn có thể tùy chỉnh các kịch bản `.sh` hoặc chạy trực tiếp `run/run_main.py` với các tham số dưới đây.

### Cấu hình cơ bản (Basic Config)
- `--model`: Tên mô hình cần sử dụng. Ví dụ: `TimeLLM`, `DLinear`, `Autoformer`.
- `--model_id`: Tên định danh cho thử nghiệm của bạn. Model checkpoint và kết quả sẽ được lưu với tên này. Ví dụ: `ETTh1_512_96`.
- `--is_training`: Đặt là `1` để huấn luyện mô hình, `0` để chỉ chạy đánh giá trên bộ test (yêu cầu đã có checkpoint).
- `--task_name`: Loại tác vụ. Thường là `long_term_forecast` hoặc `short_term_forecast`.

### Cấu hình dữ liệu (Data Loader)
- `--data`: Tên của bộ dữ liệu. Ví dụ: `ETTh1`, `ECL`, `Weather`.
- `--root_path`: Đường dẫn đến thư mục chứa file dữ liệu. Mặc định: `./dataset/`.
- `--data_path`: Tên file dữ liệu cụ thể. Ví dụ: `ETTh1.csv`.
- `--features`: Loại dự báo.
    - `M`: Multivariate -> Multivariate (dùng nhiều biến đầu vào để dự đoán nhiều biến đầu ra).
    - `S`: Univariate -> Univariate (dùng một biến đầu vào để dự đoán một biến đầu ra).
    - `MS`: Multivariate -> Univariate (dùng nhiều biến đầu vào để dự đoán một biến đầu ra).
- `--target`: Tên cột mục tiêu (target) cần dự đoán trong tác vụ `S` hoặc `MS`.
- `--freq`: Tần suất của dữ liệu, dùng để mã hóa thời gian. `h` cho hàng giờ, `d` cho hàng ngày, `t` cho hàng phút.

### Cấu hình Forecasting (Forecasting Task)
- `--seq_len`: Độ dài chuỗi đầu vào (input sequence length) đưa vào encoder.
- `--label_len`: Độ dài của phần "start token" đưa vào decoder.
- `--pred_len`: Độ dài chuỗi cần dự đoán (prediction sequence length).

### Cấu hình Mô hình (Model Definition)
- `--enc_in` / `--dec_in` / `--c_out`: Số lượng biến đầu vào cho encoder/decoder và số lượng biến đầu ra. Tương ứng với số cột features trong dữ liệu.
- `--d_model`: Kích thước chiều của mô hình (dimension of model).
- `--d_ff`: Kích thước chiều của lớp Feed-Forward.
- `--n_heads`: Số lượng "attention heads" trong mô hình Transformer-based.
- `--e_layers`: Số lớp (layers) trong Encoder.
- `--d_layers`: Số lớp (layers) trong Decoder.
- `--llm_layers` (Dành cho TimeLLM): Số lớp của mô hình ngôn ngữ lớn (LLM) được sử dụng.

### Cấu hình Tối ưu hóa (Optimization)
- `--train_epochs`: Số lượng epochs để huấn luyện.
- `--batch_size`: Kích thước của một batch dữ liệu.
- `--learning_rate`: Tốc độ học của optimizer.
- `--loss`: Hàm loss để sử dụng. Ví dụ: `MSE`, `SMAPE`.
- `--lradj`: Chiến lược điều chỉnh learning rate trong quá trình huấn luyện.
- `--patience`: Số epochs để chờ trước khi dừng sớm (early stopping) nếu validation loss không cải thiện.
