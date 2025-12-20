# Hướng dẫn Thiết kế Kịch bản Shell mới

Tài liệu này giải thích cách bạn có thể tự thiết kế một kịch bản (`.sh`) mới để chạy các thử nghiệm với những cấu hình và bộ tham số riêng.

## Giới thiệu

Các kịch bản shell trong thư mục `scripts/` là cách tiêu chuẩn và có thể tái lập để chạy các thử nghiệm. Việc gói các tham số vào một file shell giúp:
- Dễ dàng theo dõi các thử nghiệm đã chạy.
- Đảm bảo tính nhất quán khi chạy lại.
- Dễ dàng chia sẻ cấu hình thử nghiệm.

Có hai loại kịch bản chính:
- `scripts/single_gpu/`: Dành cho việc chạy trên một GPU.
- `scripts/multi_gpu/`: Dành cho việc chạy trên nhiều GPU, yêu cầu `accelerate`.

---

## 1. Cấu trúc của một kịch bản

Một kịch bản `.sh` trong dự án này thường có 2 phần chính:

#### Phần 1: Khai báo biến (Variable Declaration)

Phần đầu của file dùng để khai báo các biến chứa các siêu tham số (hyperparameters) quan trọng. Việc tập trung các biến ở đầu file giúp dễ dàng đọc và chỉnh sửa.

**Ví dụ từ `scripts/single_gpu/TimeLLM_ETTh1.sh`:**
```bash
export CUDA_VISIBLE_DEVICES=0 # Chỉ định GPU để sử dụng

model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

batch_size=24
d_model=32
d_ff=128

comment='TimeLLM-ETTh1'
```

#### Phần 2: Lệnh thực thi (Execution Command)

Phần này sẽ gọi file Python thực thi (`run/run_main.py`, `run/run_m4.py`, ...) và truyền các biến đã khai báo ở trên vào các tham số của dòng lệnh.

**Ví dụ (tiếp theo):**
```bash
python run/run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
```
- **Lưu ý:** `$` được dùng để tham chiếu đến giá trị của biến (ví dụ: `$model_name` sẽ được thay thế bằng `TimeLLM`).

---

## 2. Các bước tạo một kịch bản mới

Cách dễ nhất để tạo một kịch bản mới là sao chép và chỉnh sửa một file đã có.

#### Bước 1: Chọn một kịch bản mẫu

Tìm một file `.sh` trong `scripts/single_gpu` (hoặc `multi_gpu`) có cấu hình gần nhất với thử nghiệm bạn muốn chạy. Ví dụ, nếu bạn muốn chạy trên bộ dữ liệu `Weather`, hãy bắt đầu từ file `TimeLLM_Weather.sh`.

#### Bước 2: Sao chép và Đổi tên file

Sao chép file mẫu và đặt cho nó một cái tên có ý nghĩa, theo quy ước: `<TênMôHình>_<TênDataset>_<ThamSốĐặcBiệtNếuCó>.sh`.

**Ví dụ:**
```bash
# Sao chép file
cp scripts/single_gpu/TimeLLM_Weather.sh scripts/single_gpu/DLinear_Weather.sh
```

#### Bước 3: Chỉnh sửa các biến

Mở file mới và thay đổi các giá trị của các biến ở đầu file cho phù hợp với thử nghiệm của bạn.

**Ví dụ:** Trong file `DLinear_Weather.sh` mới, bạn có thể thay đổi:
```bash
# Thay đổi model_name
model_name=DLinear

# Giảm số epochs để chạy nhanh hơn
train_epochs=5 

# Thay đổi learning rate
learning_rate=0.005 

# DLinear không dùng llama_layers, nhưng để đó cũng không sao
# vì nó chỉ được dùng nếu model là TimeLLM
llama_layers=32 
```

#### Bước 4: Chỉnh sửa lệnh thực thi

Xem lại phần lệnh thực thi để đảm bảo các tham số cố định (không được định nghĩa bằng biến) là chính xác.
- **`--model $model_name`**: Đảm bảo bạn đã thay đổi biến `model_name`.
- **`--model_id`**: Đặt một ID mới để lưu checkpoint và kết quả. Ví dụ: `DLinear_Weather_96`.
- **`--pred_len`, `--seq_len`**: Chỉnh sửa độ dài dự đoán và độ dài đầu vào nếu cần.
- **`--enc_in`, `--dec_in`, `--c_out`**: Các tham số này phụ thuộc vào số feature của dataset. Hãy đảm bảo chúng là đúng (ví dụ, Weather có 21 features).

---

## 3. Ví dụ cụ thể

Hãy tạo một kịch bản để chạy mô hình `DLinear` trên `Weather` với độ dài dự đoán là `192`.

1.  **Sao chép:** `cp scripts/single_gpu/TimeLLM_Weather.sh scripts/single_gpu/DLinear_Weather_192.sh`
2.  **Mở file** `scripts/single_gpu/DLinear_Weather_192.sh` và chỉnh sửa.

**Nội dung gốc (một phần):**
```bash
model_name=TimeLLM
train_epochs=10
learning_rate=0.01

python run/run_main.py \
  --model_id weather_512_96 \
  --model $model_name \
  --pred_len 96 \
  # ...
```

**Nội dung sau khi chỉnh sửa:**
```bash
model_name=DLinear  # <-- Thay đổi
train_epochs=20     # <-- Tăng epochs
learning_rate=0.001 # <-- Giảm learning rate

python run/run_main.py \
  --model_id DLinear_weather_512_192 \ # <-- Thay đổi model_id
  --model $model_name \
  --pred_len 192 \ # <-- Thay đổi độ dài dự đoán
  # ... các tham số khác giữ nguyên hoặc chỉnh sửa nếu cần
```

---

## 4. Lưu ý quan trọng

- **Chạy từ thư mục gốc:** Luôn luôn chạy các kịch bản shell từ thư mục gốc của dự án để các đường dẫn tương đối (như `./dataset/`) được nhận diện đúng.
- **Kiểm tra đường dẫn:** Đảm bảo tham số `--root_path` và `--data_path` trỏ đúng đến vị trí dữ liệu của bạn.
- **Multi-GPU:** Nếu bạn tạo kịch bản cho `multi_gpu`, hãy giữ nguyên phần `accelerate launch`. Đảm bảo bạn đã cấu hình accelerate bằng lệnh `accelerate config`.
