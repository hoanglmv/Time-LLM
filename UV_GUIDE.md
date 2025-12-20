# Hướng dẫn cài đặt và sử dụng `uv` cho dự án Time-LLM

`uv` là một công cụ quản lý gói và môi trường ảo cho Python, được thiết kế để có tốc độ rất nhanh. Sử dụng `uv` có thể giúp bạn thiết lập môi trường cho dự án này một cách nhanh chóng, đặc biệt là khi làm việc trên các máy chủ từ xa hoặc GPU thuê.

## 1. Cài đặt `uv`

Bạn có thể cài đặt `uv` bằng nhiều cách khác nhau. Dưới đây là các lệnh phổ biến cho các hệ điều hành khác nhau.

**macOS / Linux (dùng curl):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (dùng PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sau khi cài đặt, hãy chắc chắn rằng `uv` đã được thêm vào `PATH` của hệ thống. Bạn có thể cần phải khởi động lại terminal hoặc cửa sổ dòng lệnh.

Kiểm tra cài đặt thành công bằng lệnh:
```bash
uv --version
```

## 2. Thiết lập môi trường ảo và cài đặt dependencies

Dự án này sử dụng `pyproject.toml` để quản lý các gói phụ thuộc. Bạn có thể sử dụng `uv` để tạo môi trường ảo và đồng bộ hóa nó với các gói được chỉ định chỉ bằng một lệnh.

Mở terminal hoặc command prompt tại thư mục gốc của dự án (cùng cấp với file `pyproject.toml`) và chạy lệnh sau:

```bash
# Lệnh này sẽ tự động:
# 1. Tạo một môi trường ảo trong thư mục .venv nếu chưa có.
# 2. Cài đặt/gỡ bỏ các gói để môi trường khớp chính xác với file pyproject.toml.
uv sync
```
Lệnh `uv sync` là cách nhanh chóng và đáng tin cậy để đảm bảo môi trường của bạn luôn được cập nhật và chính xác theo định nghĩa của dự án.

## 3. Chạy các kịch bản (scripts) của dự án

Sau khi môi trường đã được thiết lập, bạn có thể chạy các kịch bản của dự án (ví dụ: `run_fast.py` hoặc các kịch bản trong `scripts/`) bằng cách kích hoạt môi trường ảo.

**Kích hoạt môi trường ảo:**

*   **macOS / Linux:**
    ```bash
    source .venv/bin/activate
    ```
*   **Windows (Command Prompt):**
    ```bash
    .venv\Scripts\activate
    ```
*   **Windows (PowerShell):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

**Chạy kịch bản:**

Khi môi trường đã được kích hoạt, bạn có thể chạy các file Python như bình thường:
```bash
# Ví dụ chạy thử nghiệm nhanh
python run/run_fast.py

# Hoặc chạy một kịch bản huấn luyện cụ thể
# (trên Linux/macOS)
bash scripts/single_gpu/TimeLLM_ETTh1.sh

# (trên Windows, bạn có thể cần chạy lệnh python trực tiếp)
# python run/run_main.py --is_training 1 ... (các tham số tương ứng)
```

Bằng cách này, bạn có thể đảm bảo rằng dự án luôn chạy với đúng các phiên bản thư viện đã được chỉ định, giúp việc triển khai trên các hệ thống khác nhau trở nên dễ dàng và nhất quán hơn.
